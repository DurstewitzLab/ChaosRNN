from torch import optim
from torch import nn
import torch as tc
from bptt import models
from bptt import regularization
from bptt import saving
from timeit import default_timer as timer

class BPTT:
    """
    Train a model with sparsely forced BPTT.
    """
    def __init__(self, args, data_set, writer, save_path):
        self.n_epochs = args.n_epochs
        self.data_set = data_set
        self.model = models.Model(args, data_set)
        self.optimizer = optim.Adam(self.model.get_parameters(), args.learning_rate)
        self.gradient_clipping = args.gradient_clipping
        self.writer = writer
        self.use_reg = args.use_reg
        self.regularizer = regularization.Regularizer(args)
        self.saver = saving.Saver(writer, save_path, args, self.data_set, self.regularizer)
        self.n = args.n_interleave
        self.loss_fn = nn.MSELoss()
        self.windowing = args.windowing
        self.random = args.random
        self.deltaTau = args.deltaTau
        ###
        self.annealing = True
    def compute_loss(self, pred: tc.Tensor, target: tc.Tensor) -> tc.Tensor:
        '''
        Compute Loss w/ optional MAR loss.
        '''
        loss = .0
        loss += self.loss_fn(pred, target)
        loss_reg = 0
        if self.use_reg:
            lat_model_parameters = self.model.latent_model.get_latent_parameters()
            loss_reg = self.regularizer.loss(lat_model_parameters)

        return loss+loss_reg

    def train(self):
        if self.windowing:
            print("Windowing active")
        if self.random:
            print("Randomizing tau active")
        self.model.train()
        n = self.n
        self.loss_fn = nn.MSELoss()
        norm_spafo = tc.zeros(self.n_epochs+1)
        norm_bptt = tc.zeros(self.n_epochs+1)

        for epoch in range(1, self.n_epochs + 1):
            # measure time
            T_start = timer()

            for idx, (inp, target) in enumerate(self.data_set.get_rand_dataloader()):

                self.optimizer.zero_grad()
                if self.windowing:
                    pred = self.model(inp, n,z0 = tc.zeros((inp.shape[0],self.model.latent_model.d_z)))
                else:
                    if self.random:
                        lower_bound = self.n-self.deltaTau
                        upper_bound = self.n +self.deltaTau +1
                        if lower_bound<1: lower_bound = 1
                        n = int(tc.randint(lower_bound,upper_bound,(1,)))
                        if idx==1:
                            print("Tau for this batch = "+str(n))
                    if self.annealing:
                        n = int((self.n-1)/self.n_epochs*epoch)+1

                    pred,z = self.model(inp, n,z0= None,grads=True)
                    # if idx ==0:
                    #     A,W,h = self.model.latent_model.get_latent_parameters()
                    #     dx = self.model.latent_model.d_x
                    #     dz = self.model.latent_model.d_z
                    #
                    #     _,z = self.model.generate_free_trajectory(inp[0],3*10**3)
                    #     d = tc.zeros_like(z[-1])
                    #     d[z[-1] > 0] = 1
                    #     prod = tc.diag(A) + tc.matmul(W, tc.diag(d))
                    #     for i in range(1, z.shape[0]):
                    #         d = tc.zeros_like(z[ - i])
                    #         d[z[ - i] > 0] = 1
                    #         new = tc.diag(A) + tc.matmul(W, tc.diag(d))
                    #         prod = tc.matmul(prod, new)
                    #         if i == n:
                    #             norm_spafo[epoch]= tc.linalg.norm(prod, ord=2)
                    #     try:
                    #         norm_bptt[epoch] = tc.linalg.norm(prod, ord=2)
                    #     except RuntimeError:
                    #         norm_bptt[epoch] = float('nan')
                    #     print(norm_bptt[epoch],norm_spafo[epoch])

                loss = self.compute_loss(pred, target)

                loss.backward()

                if self.gradient_clipping is not None:
                    nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.gradient_clipping)
                    #nn.utils.clip_grad_value_(parameters=self.model.parameters(), clip_value=self.gradient_clipping)

                self.optimizer.step()
            T_end = timer()
            print(f"Epoch {epoch} took {round(T_end-T_start, 2)}s -----> Loss = {loss.item()}")

            if epoch % (self.n_epochs / 5) == 0:
                self.saver.epoch_save(self.model, epoch)

        #tc.save(norm_spafo, "chaos_norm_spafo2.pt")
        #tc.save(norm_bptt, "chaos_norm_bptt2.pt")