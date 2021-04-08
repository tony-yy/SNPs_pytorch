import collections
import numpy as np
import torch
from tqdm import tqdm
from scipy import linalg

SNPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "trgt_y", "num_tot_pnts", "num_cxt_pnts", "hyperparams")
)

def custom_matrix_cholesky(A):
    L = torch.zeros_like(A)

    for i in range(A.shape[-1]):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s = s + L[i,k] * L[j,k]

            L[i,j] = torch.sqrt(A[i,i] - s) if (i == j) else \
                      (1.0 / L[j,j] * (A[i,j] - s))
    return L

def cholesky_with_exception_handle(x, upper=False, force_iterative=False):
    # ****** OBSOLETE ******
    success = False
    if not force_iterative:
        try:
            results = torch.cholesky(x, upper=upper)
            success = True
        except RuntimeError as e:
            print(str(e)+", bypassing exception by custom cholesky impl.")

    if not success:
        # fall back to operating on each element separately
        results_list = []
        x_batched = x.reshape(-1, x.shape[-2], x.shape[-1])
        print("batch_id: ", end="")
        for batch_idx in range(x_batched.shape[0]):
            try:
                result = torch.cholesky(x_batched[batch_idx, :, :], upper=upper)
            except RuntimeError:
                result = custom_matrix_cholesky(x_batched[batch_idx, :, :])
            results_list.append(result)
            print(batch_idx, end=" ")
        results = torch.cat(results_list, dim=0).reshape(*x.shape)
        print("\nbatch-elem-wise cholesky decomposition impl done.")

    return results

class GPCurvesReader(object):
    def __init__(self, batch_size, max_num_cxt, 
                x_size=1, y_size=1,
                len_seq=10, len_given=5, len_gen=10, 
                l1_min=0.7, l1_max=1.2, l1_vel=0.05, 
                sigma_min=1.0, sigma_max=1.6, sigma_vel=0.05,
                testing=False, task_type=1):
        # task_type: refer to task(a), task(b), task(c) from SNP paper
        # remove temporal argm from original generator, since no longer considering NP structure
        self._batch_size = batch_size
        self._max_num_cxt = max_num_cxt
        # x_size and y_size are both 1 since the data is 1d data
        self._x_size = x_size
        self._y_size = y_size
        self._len_seq = len_seq
        self._len_given = len_given
        self._len_gen = len_gen
        self._l1_min = l1_min
        self._l1_max = l1_max
        self._l1_vel = l1_vel
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._sigma_vel = sigma_vel
        self._task_type = task_type
        self._testing = testing

        self._noise_factor = 0.1
    
    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=5e-2):
        # **increase the default sigma_noise to avoid singular kernel issue***

        # xtata: [batch_size, num_tot_pnts, x_size]
        num_tot_pnts = xdata.shape[1]

        xdata1 = torch.unsqueeze(xdata, dim=1)
        xdata2 = torch.unsqueeze(xdata, dim=2)
        diff = xdata1 - xdata2
        norm = torch.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])
        norm = torch.sum(norm, dim=-1, keepdim=False)

        kernel = torch.square(sigma_f)[:, :, None, None] * torch.exp(-0.5*norm)
        kernel += (sigma_noise**2) * torch.eye(num_tot_pnts)

        return kernel
    
    def generate_curves(self, l1, sigma_f, 
                        num_cxt=3, y_val_base=None, seed=None):
        if seed:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
        if self._testing:
            num_tot_pnts = 400
            num_trgts = num_tot_pnts
            x_vals = torch.unsqueeze(torch.range(-4., 4., 1./50, dtype=torch.float32)[:-1], dim=0).repeat([self._batch_size, 1])
        else:
            num_tot_pnts = 100
            maxval = self._max_num_cxt - num_cxt + 1
            num_trgts = torch.randint(low=1, high=int(maxval), size=[1], dtype=torch.int32)
            x_vals = torch.unsqueeze(torch.range(-4., 4., 1./12.5, dtype=torch.float32)[:-1], dim=0).repeat([self._batch_size, 1])
        x_vals = torch.unsqueeze(x_vals, dim=-1)
        # kernel: [batch_size, y_size, num_tot_pnts, num_tot_pnts]
        kernel = self._gaussian_kernel(x_vals, l1, sigma_f)
        cholesky = torch.cholesky(kernel.type(torch.float64)).type(torch.float32)

        # sample a curve
        # y_vals: [batch_size, num_tot_pnts, y_size]
        y_vals = torch.matmul(cholesky, y_val_base)
        y_vals = torch.squeeze(y_vals, dim=3).permute([0, 2, 1])

        if self._testing:
            trgt_x = x_vals
            trgt_y = y_vals

            # select observations
            idx = torch.range(0, num_trgts)[:-1].type(torch.int64)
            idx = idx[torch.randperm(idx.shape[0])]
            cxt_x = torch.index_select(x_vals, 1, idx[:num_cxt])
            cxt_y = torch.index_select(y_vals, 1, idx[:num_cxt])
        else:
            idx = torch.range(0, num_tot_pnts)[:-1].type(torch.int64)
            trgt_x = torch.index_select(x_vals, 1, idx[:num_trgts+num_cxt])
            trgt_y = torch.index_select(y_vals, 1, idx[:num_trgts+num_cxt])

            # select observations
            cxt_x = torch.index_select(x_vals, 1, idx[:num_cxt])
            cxt_y = torch.index_select(y_vals, 1, idx[:num_cxt])
        
        query = ((cxt_x, cxt_y), trgt_x)

        return SNPRegressionDescription(
            query=query,
            trgt_y=trgt_y,
            num_tot_pnts=trgt_x.shape[1],
            num_cxt_pnts=num_cxt,
            hyperparams=torch.Tensor([0])
        )

    def generate_temporal_curves(self, seed=None):
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # set kernel params
        l1 = self._l1_min * torch.ones([self._batch_size, self._y_size, self._x_size]) \
            + torch.rand([self._batch_size, self._y_size, self._x_size])*(self._l1_max-self._l1_min)
        sigma_f = self._sigma_min * torch.ones([self._batch_size, self._y_size]) \
                + torch.rand([self._batch_size, self._y_size])*(self._sigma_max-self._sigma_min)
        l1_vel = -1 * self._l1_min * torch.ones([self._batch_size, self._y_size, self._x_size]) \
            + torch.rand([self._batch_size, self._y_size, self._x_size])*(2*self._l1_vel)
        sigma_f_vel = -1 * self._sigma_min * torch.ones([self._batch_size, self._y_size]) \
                + torch.rand([self._batch_size, self._y_size])*(2*self._sigma_vel)
        
        if self._testing:
            num_tot_pnts = 400
        else:
            num_tot_pnts = 100
        y_val_base = torch.zeros([self._batch_size, self._y_size, num_tot_pnts, 1]).normal_()

        curve_list = []
        if self._task_type==2 or self._task_type==3:
            idx = torch.range(0, self._len_seq)[:-1].type(torch.int64)
            idx = idx[torch.randperm(idx.shape[0])][:self._len_given]
        
        for t in tqdm(range(self._len_seq), desc="main temporal GP context generation"):
            if seed:
                seed_t = seed * t
                torch.manual_seed(seed_t)
                torch.cuda.manual_seed(seed_t)
            else:
                seed_t = seed
            if self._task_type == 1:
                # 10 context points, 20 total points
                if t < self._len_given:
                    num_cxt = torch.randint(low=5, high=self._max_num_cxt, size=[1], dtype=torch.int32)
                else:
                    num_cxt = torch.Tensor([0]).type(torch.int32)
            if self._task_type == 2:
                # sprinkled context points
                nc_cond = torch.where(torch.eq(idx, t))[0]
                num_cxt = torch.Tensor([0]).type(torch.int32) if nc_cond.shape[0]==0 \
                            else torch.randint(low=5, high=self._max_num_cxt, size=[1], dtype=torch.int32)
            if self._task_type == 3:
                # long term tracking
                nc_cond = torch.where(torch.eq(idx, t))[0]
                num_cxt = torch.Tensor([0]).type(torch.int32) if nc_cond.shape[0]==0 \
                            else torch.Tensor([1]).type(torch.int32)
            curve_list.append(self.generate_curves(l1, sigma_f, num_cxt,
                                                    y_val_base, seed_t))
            vel_noise = l1_vel * self._noise_factor \
                        * torch.zeros([self._batch_size, self._y_size, self._x_size]).normal_()
            l1 += l1_vel + vel_noise
            vel_noise = sigma_f_vel * self._noise_factor \
                        * torch.zeros([self._batch_size, self._x_size]).normal_()
            sigma_f += sigma_f_vel + vel_noise
        
        if self._testing:
            for t in tqdm(range(self._len_seq, self._len_seq+self._len_gen), desc="testing temporal GP targets generation"):
                if seed:
                    seed_t = seed * t
                    torch.manual_seed(seed_t)
                    torch.cuda.manual_seed(seed_t)
                else:
                    seed_t = seed
                num_cxt = torch.Tensor([0]).type(torch.int32)
                curve_list.append(self.generate_curves(l1, sigma_f, num_cxt,
                                                    y_val_base, seed_t))
                vel_noise = l1_vel * self._noise_factor \
                        * torch.zeros([self._batch_size, self._y_size, self._x_size]).normal_()
                l1 += l1_vel + vel_noise
                vel_noise = sigma_f_vel * self._noise_factor \
                            * torch.zeros([self._batch_size, self._x_size]).normal_()
                sigma_f += sigma_f_vel + vel_noise
        
        cxt_x_list, cxt_y_list = [], []
        trgt_x_list, trgt_y_list = [], []
        num_total_points_list = []
        num_cxt_pnts_list = []

        for t in range(len(curve_list)):
            (cxt_x, cxt_y), trgt_x = curve_list[t].query
            trgt_y = curve_list[t].trgt_y
            num_total_points_list.append(curve_list[t].num_tot_pnts)
            num_cxt_pnts_list.append(curve_list[t].num_cxt_pnts)
            cxt_x_list.append(cxt_x)
            cxt_y_list.append(cxt_y)
            trgt_x_list.append(trgt_x)
            trgt_y_list.append(trgt_y)

        query = ((cxt_x_list, cxt_y_list), trgt_x_list)
        return SNPRegressionDescription(
            query=query,
            trgt_y=trgt_y_list,
            num_tot_pnts=num_total_points_list,
            num_cxt_pnts=num_cxt_pnts_list,
            hyperparams=[torch.Tensor([0])])         

if __name__=="__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    batch_size = 16
    max_num_cxt = 500   # max cxt size at each timestep
    len_seq, len_given, len_gen = 20, 10, 10
    l1_min, l1_max, l1_vel = 0.7, 1.2, 0.03
    sigma_min, sigma_max, sigma_vel = 1.0, 1.6, 0.05
    task_type = 1

    # training dataset
    dataset_train = GPCurvesReader(
        batch_size=batch_size, max_num_cxt=max_num_cxt,
        len_seq=len_seq, len_given=len_given, len_gen=len_gen,
        l1_min=l1_min, l1_max=l1_max, l1_vel=l1_vel,
        sigma_min=sigma_min, sigma_max=sigma_max, sigma_vel=sigma_vel,
        task_type=task_type
    )
    data_train = dataset_train.generate_temporal_curves()
    
    # testing dataset
    dataset_test = GPCurvesReader(
        batch_size=batch_size, max_num_cxt=max_num_cxt,
        len_seq=len_seq, len_given=len_given, len_gen=len_gen,
        l1_min=l1_min, l1_max=l1_max, l1_vel=l1_vel,
        sigma_min=sigma_min, sigma_max=sigma_max, sigma_vel=sigma_vel,
        testing=True, task_type=task_type
    )
    data_test = dataset_test.generate_temporal_curves(seed=1234)

    

