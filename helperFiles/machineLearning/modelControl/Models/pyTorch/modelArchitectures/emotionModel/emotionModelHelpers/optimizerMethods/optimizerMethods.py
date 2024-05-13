import torch.optim as optim
import transformers


class optimizerMethods:

    def addOptimizer(self, submodel, signalEncoderModel, autoencoderModel, signalMappingModel, sharedEmotionModel, specificEmotionModel):
        modelParams = [
            # Specify the model parameters for the signal encoding.
            {'params': signalEncoderModel.parameters(), 'weight_decay': 1E-10, 'lr': 2E-4}]  # Empirically: 1E-10 < weight_decay < 1E-6; 1E-4 < lr < 5E-4
        if submodel in ["autoencoder", "emotionPrediction"]:
            modelParams.append(
                # Specify the model parameters for the autoencoder.
                {'params': autoencoderModel.parameters(), 'weight_decay': 1E-8, 'lr': 2E-4})
        if submodel == "emotionPrediction":
            modelParams.extend([
                # Specify the model parameters for the signal mapping.
                {'params': signalMappingModel.parameters(), 'weight_decay': 1E-6, 'lr': 1E-4},

                # Specify the model parameters for the feature extraction.
                {'params': sharedEmotionModel.extractCommonFeatures.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4},

                # Specify the model parameters for the human activity recognition.
                {'params': sharedEmotionModel.extractActivityFeatures.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4},
                {'params': specificEmotionModel.classifyHumanActivity.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4},

                # Specify the model parameters for the emotion prediction.
                {'params': sharedEmotionModel.predictBasicEmotions.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4},
                {'params': specificEmotionModel.predictUserEmotions.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4},
                {'params': specificEmotionModel.predictComplexEmotions.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4}])

        # Set the optimizer.
        optimizer = self.setOptimizer(modelParams, lr=1E-4, weight_decay=1E-6, submodel=submodel)

        # Set the learning rate scheduler.
        scheduler = self.getLearningRateScheduler(submodel, optimizer)

        return optimizer, scheduler

    def setOptimizer(self, params, lr, weight_decay, submodel):
        if submodel == "signalEncoder":
            # Observations on properties:
            #     Momentum is not good (Used value of 0.9)
            # Observations on encoding:
            #     No encoding structure: Rprop, Adamax, RMSprop
            #     No-Noisy encoding structure: Adam, NAdam, RAdam
            #     Noisy encoding structure: AdamW
            #     Okay encoding structure: SGD, ASGD, Adadelta
            # Observations on reconstruction:
            #     No reconstruction: Adadelta, RMSprop
            #     No reconstruction, but trying: ASGD, Adamax
            #     No-Noisy reconstruction: SGD
            #     Noisy reconstruction: RAdam
            #     Okay reconstruction: AdamW, Adam, NAdam, Rprop
            return self.getOptimizer(optimizerType="RMSprop", params=params, lr=lr, weight_decay=weight_decay, momentum=0)
        elif submodel == "autoencoder":
            return optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False, maximize=False)
        elif submodel == "emotionPrediction":
            # adam and RAdam are okay, AdamW is a bit better (best?); NAdam is also a bit better
            return optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False, maximize=False)
        else:
            assert False, "No optimizer initialized"

    @staticmethod
    def getLearningRateScheduler(submodel, optimizer):
        # Options:
        # Slow ramp up: transformers.get_constant_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=30)
        # Cosine waveform: optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-8, last_epoch=-1, verbose=False)
        # Reduce on plateau (need further editing of loop): optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        # Defined lambda function: optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_function); lambda_function = lambda epoch: (epoch/50) if epoch < -1 else 1
        # torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.3333333333333333, end_factor=1.0, total_iters=5, last_epoch=-1)

        # Train the autoencoder
        if submodel == "signalEncoder":
            return transformers.get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=5)
        elif submodel == "autoencoder":
            return transformers.get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=5)
        elif submodel == "emotionPrediction":
            return transformers.get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=5)
        else:
            assert False, "No model initialized"

    @staticmethod
    def getOptimizer(optimizerType, params, lr, weight_decay, momentum=0.9):
        # General guidelines:
        #     Common WD values: 1E-2 to 1E-6
        #     Common LR values: 1E-6 to 1

        if optimizerType == 'Adadelta':
            # Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
            # Use it when you don’t want to manually tune the learning rate.
            return optim.Adadelta(params, lr=lr, rho=0.9, eps=1e-06, weight_decay=weight_decay)
        elif optimizerType == 'Adagrad':
            # Adagrad adapts the learning rates based on the parameters. It performs well with sparse data.
            # Use it if you are dealing with sparse features or in NLP tasks. Not compatible with GPU?!?
            return optim.Adagrad(params, lr=lr, lr_decay=0, weight_decay=weight_decay, initial_accumulator_value=0, eps=1e-10)
        elif optimizerType == 'Adam':
            # Adam is a first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates.
            # It's broadly used and suitable for most problems without much hyperparameter tuning.
            return optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False, maximize=False)
        elif optimizerType == 'AdamW':
            # AdamW modifies the way Adam implements weight decay, decoupling it from the gradient updates, leading to a more effective use of L2 regularization.
            # Use when regularization is a priority and particularly when fine-tuning pre-trained models.
            return optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False, maximize=False)
        elif optimizerType == 'NAdam':
            # NAdam combines Adam with Nesterov momentum, aiming to combine the benefits of Nesterov accelerated gradient and Adam.
            # Use in deep architectures where fine control over convergence is needed.
            return optim.NAdam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, momentum_decay=0.004, decoupled_weight_decay=True)
        elif optimizerType == 'RAdam':
            # RAdam (Rectified Adam) is an Adam variant that introduces a term to rectify the variance of the adaptive learning rate.
            # Use it when facing unstable or poor training results with Adam, especially in smaller sample sizes.
            return optim.RAdam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, decoupled_weight_decay=True)
        elif optimizerType == 'Adamax':
            # Adamax is a variant of Adam based on the infinity norm, proposed as a more stable alternative.
            # Suitable for embeddings and sparse gradients.
            return optim.Adamax(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        elif optimizerType == 'ASGD':
            # ASGD (Averaged Stochastic Gradient Descent) is used when you require robustness over a large number of epochs.
            # Suitable for larger-scale and less well-behaved problems; often used in place of SGD when training for a very long time.
            return optim.ASGD(params, lr=lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=weight_decay)
        elif optimizerType == 'LBFGS':
            # LBFGS is an optimizer that approximates the Broyden–Fletcher–Goldfarb–Shanno algorithm, which is a quasi-Newton method.
            # Use it for small datasets where the exact second-order Hessian matrix computation is possible. Maybe cant use optimizer.step()??
            return optim.LBFGS(params, lr=lr, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
        elif optimizerType == 'RMSprop':
            # RMSprop is an adaptive learning rate method designed to solve Adagrad's radically diminishing learning rates.
            # It is well-suited to handle non-stationary objectives as in training neural networks.
            return optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-08, weight_decay=weight_decay, momentum=momentum, centered=False)
        elif optimizerType == 'Rprop':
            # Rprop (Resilient Propagation) uses only the signs of the gradients, disregarding their magnitude.
            # Suitable for batch training, where the robustness of noisy gradients and the size of updates matters.
            return optim.Rprop(params, lr=lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        elif optimizerType == 'SGD':
            # SGD (Stochastic Gradient Descent) is simple yet effective, suitable for large datasets.
            # Use with momentum for non-convex optimization; ideal for most cases unless complexities require adaptive learning rates.
            return optim.SGD(params, lr=lr, momentum=momentum, dampening=0, weight_decay=weight_decay, nesterov=True)
        else:
            assert False, "No optimizer initialized"
