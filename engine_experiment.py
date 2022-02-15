import utils
import engine
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix

class experiment():
    def __init__(self, fold_idx, task):
        self.fold_idx = fold_idx
        self.task = task

        # Learning schedules
        self.num_epochs = 200 # 100
        self.num_batches = 5
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=1000,
                                                                 decay_rate=.96, staircase=False)  # init_lr: 1e-3

        # Loss control hyperparameter
        self.alpha_rec = .7  # reconstruction
        self.alpha_gen = .5  # generation
        self.alpha_dis = 1  # discrimination
        self.alpha_clf = 1  # classification
        self.alpha_reg = .7  # regression

    def training(self):
        print(f'Start Training, Fold {self.fold_idx}')

        # Load dataset
        X_MRI_train, E_SNP_train, C_demo_train, Y_train, S_train, \
        X_MRI_test, E_SNP_test, C_demo_test, Y_test, S_test = utils.load_dataset(self.fold_idx, self.task)
        N_o = Y_train.shape[-1]

        # Call optimizers
        opt_rec = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_gen = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_dis = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_clf = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_reg = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Call ENGINE framework
        model = engine.engine(N_o=N_o)

        num_iters = int(Y_train.shape[0]/self.num_batches)

        report = []

        for epoch in range(self.num_epochs):
            L_rec_per_epoch = 0
            L_gen_per_epoch = 0
            L_dis_per_epoch = 0
            L_clf_per_epoch = 0
            L_reg_per_epoch = 0

            # Randomize the training dataset
            rand_idx = np.random.permutation(Y_train.shape[0])
            X_MRI_train = X_MRI_train[rand_idx, ...]
            E_SNP_train = E_SNP_train[rand_idx, ...]
            C_demo_train = C_demo_train[rand_idx, ...]
            Y_train = Y_train[rand_idx, ...]
            S_train = S_train[rand_idx, ...]

            for batch in range(num_iters):
                # Sample a minibatch
                xb_MRI = X_MRI_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]
                eb_SNP = E_SNP_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(np.float32)
                cb_demo = C_demo_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]
                yb_clf = Y_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(np.float32)
                sb_reg = S_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]

                # Estimate gradient and loss
                with tf.GradientTape() as tape_rec, tf.GradientTape() as tape_gen, tf.GradientTape() as tape_dis, \
                    tf.GradientTape() as tape_clf, tf.GradientTape() as tape_reg:

                    # SNP representation module
                    mu, log_sigma_square = model.encode(x_SNP=eb_SNP)
                    zb_SNP = model.reparameterize(mean=mu, logvar=log_sigma_square)
                    eb_SNP_hat_logit = model.decode(z_SNP=zb_SNP)
                    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=eb_SNP_hat_logit, labels=eb_SNP)
                    log_prob_eb_SNP_given_zb_SNP = -tf.math.reduce_sum(cross_ent, axis=1)
                    log_prob_zb_SNP = utils.log_normal_pdf(sample=zb_SNP, mean=0., logvar=0.)
                    log_q_zb_given_eb_SNP = utils.log_normal_pdf(sample=zb_SNP, mean=mu, logvar=log_sigma_square)

                    # Reconstruction loss
                    L_rec = -tf.math.reduce_mean(log_prob_eb_SNP_given_zb_SNP + log_prob_zb_SNP - log_q_zb_given_eb_SNP)
                    L_rec *= self.alpha_rec

                    # MRI-SNP association module
                    xb_MRI_fake, ab = model.generate(z_SNP=zb_SNP, c_demo=cb_demo)
                    real_output = model.discriminate(x_MRI_real_or_fake=xb_MRI)
                    fake_output = model.discriminate(x_MRI_real_or_fake=xb_MRI_fake)

                    # Least-Square GAN loss
                    L_gen = tf.keras.losses.MSE(tf.ones_like(fake_output), fake_output)
                    L_gen *= self.alpha_gen

                    L_dis = tf.keras.losses.MSE(tf.ones_like(real_output), real_output) \
                            + tf.keras.losses.MSE(tf.zeros_like(fake_output), fake_output)
                    L_dis *= self.alpha_dis

                    # Diagnostician module
                    yb_clf_hat, sb_reg_hat = model.diagnose(x_MRI=xb_MRI, a=ab, apply_logistic_activation=True)

                    # Classification loss
                    L_clf = tfa.losses.sigmoid_focal_crossentropy(yb_clf, yb_clf_hat)
                    L_clf *= self.alpha_clf

                    # Regression loss
                    L_reg = tf.keras.losses.MSE(sb_reg, sb_reg_hat)
                    L_reg *= self.alpha_reg

                # Apply gradients
                var = model.trainable_variables
                theta_Q = [var[0], var[1], var[2], var[3]]
                theta_P = [var[4], var[5], var[6], var[7]]
                theta_G = [var[8], var[9], var[10], var[11]]
                theta_D = [var[12], var[13]]
                theta_C_share = [var[14], var[15]]
                theta_C_clf = [var[16], var[17]]
                theta_C_reg = [var[18], var[19]]

                grad_rec = tape_rec.gradient(L_rec, theta_Q + theta_P)
                opt_rec.apply_gradients(zip(grad_rec, theta_Q + theta_P))
                L_rec_per_epoch += np.mean(L_rec)

                grad_gen = tape_gen.gradient(L_gen, theta_Q + theta_G)
                opt_gen.apply_gradients(zip(grad_gen, theta_Q + theta_G))
                L_gen_per_epoch += np.mean(L_gen)

                grad_dis = tape_dis.gradient(L_dis, theta_D)
                opt_dis.apply_gradients(zip(grad_dis, theta_D))
                L_dis_per_epoch += np.mean(L_dis)

                grad_clf = tape_clf.gradient(L_clf, theta_G + theta_C_share + theta_C_clf)
                opt_clf.apply_gradients(zip(grad_clf, theta_G + theta_C_share + theta_C_clf))
                L_clf_per_epoch += np.mean(L_clf)

                grad_reg = tape_reg.gradient(L_reg, theta_G + theta_C_share + theta_C_reg)
                opt_reg.apply_gradients(zip(grad_reg, theta_G + theta_C_share + theta_C_reg))
                L_reg_per_epoch += np.mean(L_reg)

            L_rec_per_epoch /= num_iters
            L_gen_per_epoch /= num_iters
            L_dis_per_epoch /= num_iters
            L_clf_per_epoch /= num_iters
            L_reg_per_epoch /= num_iters

            # Loss report
            # print(f'Epoch: {epoch + 1}, Lrec: {L_rec_per_epoch:>.4f}, Lgen: {L_gen_per_epoch:>.4f}, '
            #       f'Ldis: {L_dis_per_epoch:>.4f}, Lclf: {L_clf_per_epoch:>.4f}, Lreg: {L_reg_per_epoch:>.4f}')

        # Results
        mu, log_sigma_square = model.encode(E_SNP_test)
        Z_SNP_test = model.reparameterize(mu, log_sigma_square)
        _, A_test = model.generate(Z_SNP_test, C_demo_test)
        Y_test_hat, S_test_hat = model.diagnose(X_MRI_test, A_test, True)
        print(f'Test AUC: {roc_auc_score(Y_test, Y_test_hat):>.4f}')
        rmse = np.sqrt(mean_squared_error(S_test * 30., S_test_hat * 30.))
        print(f'Test Regression RMSE: {rmse:>.4f}')
        return

task = ['CN', 'AD'] # ['CN', 'MCI'], ['sMCI', 'pMCI'], ['CN', 'MCI', 'AD'], ['CN', 'sMCI', 'pMCI', 'AD']
for fold in range(5): # five-fold cross-validation
    exp = experiment(fold + 1, ['CN', 'AD'])
    exp.training()
