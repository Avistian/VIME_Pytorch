import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data_utils
from vime_semi import VIMESemi
from vime_self import VIMESelf


class VIME(nn.Module):
    def __init__(
        self,
        input_dim,
        alpha,
        beta,
        optimizer,
        semi_optimizer,
        K,
        p_corr,
        semi_p_corr,
        num_epochs,
        device,
        lr,
        semi_lr,
        batch_size,
        semi_batch_size,
        encoder_backbone=None,
        feature_estimator_backbone=None,
        mask_estimator_backbone=None,
        predictor_backbone=None,
    ):
        """
        Initialize the VIME class.

        Args:
            input_dim (int): Dimensionality of the input data.
            alpha (float): Weighting factor for the self-supervised loss.
            beta (float): Weighting factor for the semi-supervised loss.
            optimizer (torch.optim.Optimizer): Optimizer for the self-supervised model.
            semi_optimizer (torch.optim.Optimizer): Optimizer for the semi-supervised model.
            K (int): Number of negative samples for the semi-supervised loss.
            p_corr (float): Probability of applying random corruption to the input data for the self-supervised loss.
            semi_p_corr (float): Probability of applying random corruption to the input data for the semi-supervised loss.
            num_epochs (int): Number of training epochs.
            device (torch.device): Device to run the models on (e.g., 'cuda' or 'cpu').
            lr (float): Learning rate for the self-supervised optimizer.
            semi_lr (float): Learning rate for the semi-supervised optimizer.
            batch_size (int): Batch size for the self-supervised training.
            semi_batch_size (int): Batch size for the semi-supervised training.
            encoder_backbone (torch.nn.Module, optional): Backbone architecture for the encoder. If not provided, a default architecture will be used.
            feature_estimator_backbone (torch.nn.Module, optional): Backbone architecture for the feature estimator. If not provided, a default architecture will be used.
            mask_estimator_backbone (torch.nn.Module, optional): Backbone architecture for the mask estimator. If not provided, a default architecture will be used.
            predictor_backbone (torch.nn.Module, optional): Backbone architecture for the predictor. If not provided, a default architecture will be used.
        """
        super(VIME, self).__init__()
        self.input_dim = input_dim
        self.K = K
        self.alpha = alpha
        self.device = device
        self.lr = lr
        self.semi_lr = semi_lr
        self.num_epochs = num_epochs
        self.p_corr = p_corr
        self.semi_p_corr = semi_p_corr
        self.beta = beta
        self.batch_size = batch_size
        self.semi_batch_size = semi_batch_size

        self.encoder_backbone = (
            self._get_default_encoder_backbone(self.input_dim, [], self.input_dim)
            if encoder_backbone is None
            else encoder_backbone
        )
        self.feature_estimator_backbone = (
            self._get_default_feature_estimator_backbone(
                self.input_dim, [], self.input_dim
            )
            if feature_estimator_backbone is None
            else feature_estimator_backbone
        )
        self.mask_estimator_backbone = (
            self._get_default_mask_estimator_backbone(
                self.input_dim, [], self.input_dim
            )
            if mask_estimator_backbone is None
            else mask_estimator_backbone
        )
        self.predictor_backbone = (
            self._get_default_predictor_backbone(self.input_dim, [100, 100], 10)
            if predictor_backbone is None
            else predictor_backbone
        )
        self.VIME_self = VIMESelf(
            self.encoder_backbone,
            self.mask_estimator_backbone,
            self.feature_estimator_backbone,
        ).to(self.device)
        self.VIME_semi = VIMESemi(
            self.encoder_backbone, self.predictor_backbone, self.K
        ).to(self.device)
        self.optimizer = optimizer(self.VIME_self.parameters(), lr=self.lr)
        self.semi_optimizer = semi_optimizer(
            self.VIME_semi.parameters(), lr=self.semi_lr
        )

    def fit(self, train_loader, val_loader, unlabeled_train_arr, unlabeled_val_arr):
        """
        Fits the VIME model using the provided data.

        Parameters:
        train_loader (DataLoader): DataLoader for the labeled training data.
        val_loader (DataLoader): DataLoader for the labeled validation data.
        unlabeled_train_arr (ndarray): Array of unlabeled training data.
        unlabeled_val_arr (ndarray): Array of unlabeled validation data.
        """
        print("Preparing unlabeled data")
        X_unsup_mask = self._generate_mask(unlabeled_train_arr.shape, self.p_corr)
        m_label, X_unsup_tilde = self._generate_pretext_task(
            unlabeled_train_arr, X_unsup_mask
        )
        unlab = torch.from_numpy(unlabeled_train_arr.astype(np.float32))
        mask = torch.from_numpy(X_unsup_mask.astype(np.float32))
        m_label = torch.from_numpy(m_label.astype(np.float32))
        unlab_tilde = data_utils.TensorDataset(
            torch.from_numpy(X_unsup_tilde.astype(np.float32)), unlab, mask, m_label
        )
        unlab_tilde_loader = data_utils.DataLoader(
            unlab_tilde, batch_size=self.batch_size, shuffle=True
        )

        X_unsup_val_mask = self._generate_mask(unlabeled_val_arr.shape, self.p_corr)
        m_label_val, X_unsup_val_tilde = self._generate_pretext_task(
            unlabeled_val_arr, X_unsup_val_mask
        )
        unlab_val = torch.from_numpy(unlabeled_val_arr.astype(np.float32))
        mask_val = torch.from_numpy(X_unsup_val_mask.astype(np.float32))
        m_label_val = torch.from_numpy(m_label_val.astype(np.float32))
        unlab_tilde_val = data_utils.TensorDataset(
            torch.from_numpy(X_unsup_val_tilde.astype(np.float32)),
            unlab_val,
            mask_val,
            m_label_val,
        )
        unlab_tilde_loader_val = data_utils.DataLoader(
            unlab_tilde_val, batch_size=self.batch_size, shuffle=True
        )

        print("Training Self VIME")
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self._train_loop_self(
                unlab_tilde_loader, self.VIME_self, self._loss_fn, self.optimizer
            )
            self._test_loop_self(unlab_tilde_loader_val, self.VIME_self, self._loss_fn)

        self.VIME_semi.encoder = self.VIME_self.encoder

        X_semi_unsup_train = np.zeros(
            (unlabeled_train_arr.shape[0], self.K, unlabeled_train_arr.shape[1])
        )
        X_semi_unsup_test = np.zeros(
            (unlabeled_val_arr.shape[0], self.K, unlabeled_val_arr.shape[1])
        )

        def generate_pretext_k_examples(row, k, p_corr):
            result = np.zeros((1, k, row.shape[0]))
            for k in range(k):
                mask_semi = self._generate_mask(row.shape, p_corr)
                m_semi_label, X_semi_unsup_tilde = self._generate_pretext_task(
                    row, mask_semi
                )
                result[0, k, :] = X_semi_unsup_tilde
            return result

        X_semi_unsup_train = np.apply_along_axis(
            generate_pretext_k_examples,
            0,
            unlabeled_train_arr,
            self.K,
            self.semi_p_corr,
        )
        X_semi_unsup_test = np.apply_along_axis(
            generate_pretext_k_examples, 0, unlabeled_val_arr, self.K, self.semi_p_corr
        )

        X_semi_unsup_train = torch.from_numpy(
            X_semi_unsup_train.astype(np.float32).reshape(
                -1, X_semi_unsup_train.shape[3]
            )
        )
        X_semi_unsup_test = torch.from_numpy(
            X_semi_unsup_test.astype(np.float32).reshape(-1, X_semi_unsup_test.shape[3])
        )

        min_loss = 100000
        min_loss_iter = 0
        print("Training Semi VIME")
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self._train_loop_semi(
                train_loader,
                X_semi_unsup_train.clone(),
                self.VIME_semi,
                self._loss_fn_semi,
                self.beta,
                self.semi_optimizer,
                self.semi_batch_size,
                self.K,
            )
            loss = self._test_loop_semi(
                val_loader,
                X_semi_unsup_test.clone(),
                self.VIME_semi,
                self._loss_fn_semi,
                self.beta,
                self.semi_batch_size,
                self.K,
            )
            if min_loss > loss:
                min_loss = loss
                min_loss_iter = epoch
            if epoch - min_loss_iter > 20:
                break

    def predict(self, x):
        """
        Predicts the output for the given input.

        Parameters:
            x (array-like): The input data.

        Returns:
            array-like: The predicted output.
        """
        y_logit, y_hat = self.VIME_semi.predict(x)
        return y_hat

    def _generate_mask(self, shape, p_corr):
        """
        Generates a binary mask with the specified shape and correlation probability.

        Parameters:
            shape (tuple): The shape of the mask.
            p_corr (float): The probability of each element being 1 (correlated).

        Returns:
            numpy.ndarray: The generated binary mask.
        """
        mask = np.random.binomial(1, p_corr, shape)
        return mask

    def _generate_pretext_task(self, x, mask):
        """
        Generates a pretext task for the given input data.

        Parameters:
        x (ndarray): The input data.
        mask (ndarray): The mask indicating which elements to modify.

        Returns:
        m (ndarray): The modified elements mask.
        x_tilde (ndarray): The modified input data.
        """
        x = x.reshape(1, x.shape[0]) if x.ndim != 2 else x
        n, dim = x.shape
        x_bar = np.zeros((n, dim))
        for d in range(dim):
            x_bar[:, d] = np.random.permutation(x[:, d])
        x_tilde = (1 - mask) * x + mask * x_bar
        m = 1 * (x != x_tilde)
        return m, x_tilde

    def _loss_fn(self, x_mask, x_feature, mask, x_unlab, alpha):
        """
        Calculates the loss function for VIME.

        Parameters:
        - x_mask (torch.Tensor): The masked input data.
        - x_feature (torch.Tensor): The feature extracted from the input data.
        - mask (torch.Tensor): The mask indicating the valid elements in the input data.
        - x_unlab (torch.Tensor): The unlabaled input data.
        - alpha (float): The weight for the MSE loss term.

        Returns:
        - loss (torch.Tensor): The calculated loss value.
        """
        mse = torch.mean((x_unlab - x_feature) ** 2)
        bce = torch.mean(F.binary_cross_entropy(x_mask, mask))
        return alpha * mse + bce

    def _loss_fn_semi(self, y_pred, yu_pred, y_true, beta):
        """
        Calculates the semi-supervised loss function.

        Parameters:
            y_pred (Tensor): Predicted labels for labeled data.
            yu_pred (Tensor): Predicted labels for unlabeled data.
            y_true (Tensor): True labels for labeled data.
            beta (float): Weight for the regularization term.

        Returns:
            Tensor: The calculated loss value.
        """
        loss = F.cross_entropy(y_pred, y_true)
        loss += beta * torch.var(yu_pred, dim=1).mean()
        return loss

    def _generate_pretext_k_examples(self, row, k, p_corr):
        """
        Generates k examples for the pretext task using the given row and correlation probability.

        Args:
            row (numpy.ndarray): The input row.
            k (int): The number of examples to generate.
            p_corr (float): The probability of correlation.

        Returns:
            numpy.ndarray: A 3-dimensional array of shape (1, k, row.shape[0]) containing the generated examples.
        """
        result = np.zeros((1, k, row.shape[0]))
        for k in range(k):
            mask_semi = self._generate_mask(row.shape, p_corr)
            m_semi_label, X_semi_unsup_tilde = self._generate_pretext_task(
                row, mask_semi
            )
            result[0, k, :] = X_semi_unsup_tilde
        return result

    def _train_loop_self(self, dataloader, model, loss_fn, optimizer, report_iters=10):
        """
        Performs the training loop for self-supervised learning.

        Args:
            dataloader (torch.utils.data.DataLoader): The data loader for the training dataset.
            model (torch.nn.Module): The model to be trained.
            loss_fn (callable): The loss function to be used.
            optimizer (torch.optim.Optimizer): The optimizer for updating the model's parameters.
            report_iters (int, optional): The number of iterations between each progress report. Defaults to 10.
        """
        size = len(dataloader)
        for batch, x in enumerate(dataloader):
            # Transfer to device
            X = x[0].to(self.device)
            unlab = x[1].to(self.device)
            m_label = x[3].to(self.device)
            mask = x[2].to(self.device)
            batch_size = X.shape[0]

            # Compute prediction and loss
            x_mask, x_feature = model(X)
            loss = loss_fn(x_mask, x_feature, m_label, unlab, self.alpha)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % report_iters == 0:
                loss, current = loss.item(), batch
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _test_loop_self(self, dataloader, model, loss_fn):
        """
        Perform testing on the given dataloader using the provided model and loss function.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the test data.
            model: The model to be used for testing.
            loss_fn: The loss function to calculate the test loss.

        Returns:
            float: The average test loss.
        """
        size = len(dataloader)
        num_batches = len(dataloader)
        test_loss = 0
        batch_size = size
        with torch.no_grad():
            for x in dataloader:
                X = x[0].to(self.device)
                unlab = x[1].to(self.device)
                m_label = x[3].to(self.device)
                mask = x[2].to(self.device)
                batch_size = X.shape[0]
                x_mask, x_feature = model(X)
                test_loss += loss_fn(x_mask, x_feature, m_label, unlab, self.alpha)

        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    def _train_loop_semi(
        self,
        train_loader,
        unsup_dataset,
        model,
        loss_fn,
        beta,
        optimizer,
        batch_size,
        K,
        report_iters=10,
    ):
        """
        Performs the semi-supervised training loop.

        Args:
            train_loader (torch.utils.data.DataLoader): The data loader for the labeled training dataset.
            unsup_dataset (torch.Tensor): The unsupervised dataset.
            model (torch.nn.Module): The model to be trained.
            loss_fn (callable): The loss function.
            beta (float): The weight for the unsupervised loss.
            optimizer (torch.optim.Optimizer): The optimizer.
            batch_size (int): The batch size.
            K (int): The number of unsupervised samples per labeled sample.
            report_iters (int, optional): The number of iterations between reporting the loss. Defaults to 10.
        """
        model.train()
        size = len(train_loader.dataset)
        for batch, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            xu = unsup_dataset[
                batch * batch_size * K : (batch + 1) * batch_size * K, :
            ].to(self.device)
            y_logit_pred, yu_logit_pred, y_pred, yu_pred = model(x, xu)
            loss = self._loss_fn_semi(y_pred, yu_logit_pred, y, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % report_iters == 0:
                loss, current = loss.item(), batch
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _test_loop_semi(
        self, test_loader, unsup_dataset, model, loss_fn, beta, batch_size, K
    ):
        """
        Performs the testing loop for the semi-supervised VIME algorithm.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            unsup_dataset (torch.Tensor): Unsupervised dataset.
            model: The VIME model.
            loss_fn: Loss function for the VIME algorithm.
            beta (float): Weight for the unsupervised loss.
            batch_size (int): Batch size for testing.
            K (int): Number of unsupervised samples per input.

        Returns:
            float: Average test loss.
        """
        size_sup = len(test_loader.dataset)
        size_unsup = unsup_dataset.shape[0]
        size = min(size_sup, size_unsup)
        num_batches = len(test_loader)
        test_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                xu = unsup_dataset[i * batch_size * K : (i + 1) * batch_size * K, :].to(
                    self.device
                )
                y_logit_pred, yu_logit_pred, y_pred, yu_pred = model(x, xu)
                test_loss += self._loss_fn_semi(y_pred, yu_logit_pred, y, beta).item()

        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
        return test_loss

    def _get_default_encoder_backbone(self, input_width, hidden_width, output_width):
        """
        Returns the default encoder backbone network.

        Args:
            input_width (int): The width of the input layer.
            hidden_width (list): A list of integers representing the widths of the hidden layers.
            output_width (int): The width of the output layer.

        Returns:
            nn.Sequential: The default encoder backbone network.
        """

        def get_network():
            if len(hidden_width) == 0:
                return [nn.Linear(input_width, output_width), nn.ReLU()]
            layers = [nn.Linear(input_width, hidden_width[0]), nn.ReLU()]
            for layer_width, next_layer_width in zip(
                hidden_width[0 : len(hidden_width) - 1], hidden_width[1:]
            ):
                layers.append(nn.Linear(layer_width, next_layer_width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_width[-1], output_width))
            layers.append(nn.ReLU())
            return layers

        return nn.Sequential(*get_network())

    def _get_default_mask_estimator_backbone(
        self, input_width, hidden_width, output_width
    ):
        """
        Returns the default mask estimator backbone network.

        Args:
            input_width (int): The width of the input layer.
            hidden_width (list): A list of integers representing the widths of the hidden layers.
            output_width (int): The width of the output layer.

        Returns:
            nn.Sequential: The default mask estimator backbone network.
        """

        def get_network():
            if len(hidden_width) == 0:
                return [nn.Linear(input_width, output_width), nn.Sigmoid()]
            layers = [nn.Linear(input_width, hidden_width[0]), nn.ReLU()]
            for layer_width, next_layer_width in zip(
                hidden_width[0 : len(hidden_width) - 1], hidden_width[1:]
            ):
                layers.append(nn.Linear(layer_width, next_layer_width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_width[-1], output_width))
            layers.append(nn.Sigmoid())
            return layers

        return nn.Sequential(*get_network())

    def _get_default_feature_estimator_backbone(
        self, input_width, hidden_width, output_width
    ):
        """
        Returns the default feature estimator backbone network.

        Args:
            input_width (int): The width of the input layer.
            hidden_width (list): A list of integers representing the width of each hidden layer.
            output_width (int): The width of the output layer.

        Returns:
            nn.Sequential: The default feature estimator backbone network.
        """

        def get_network():
            if len(hidden_width) == 0:
                return [nn.Linear(input_width, output_width)]
            layers = [nn.Linear(input_width, hidden_width[0]), nn.ReLU()]
            for layer_width, next_layer_width in zip(
                hidden_width[0 : len(hidden_width) - 1], hidden_width[1:]
            ):
                layers.append(nn.Linear(layer_width, next_layer_width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_width[-1], output_width))
            return layers

        return nn.Sequential(*get_network())

    def _get_default_predictor_backbone(self, input_width, hidden_width, output_width):
        """
        Returns the default predictor backbone network.

        Args:
            input_width (int): The width of the input layer.
            hidden_width (list): A list of integers representing the width of each hidden layer.
            output_width (int): The width of the output layer.

        Returns:
            nn.Sequential: The default predictor backbone network.
        """

        def get_network():
            layers = [nn.Linear(input_width, hidden_width[0]), nn.ReLU()]
            for layer_width, next_layer_width in zip(
                hidden_width[0 : len(hidden_width) - 1], hidden_width[1:]
            ):
                layers.append(nn.Linear(layer_width, next_layer_width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_width[-1], output_width))
            return layers

        return nn.Sequential(*get_network())
