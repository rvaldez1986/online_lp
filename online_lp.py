import numpy as np

class OnlineOptimization:
    def __init__(self, cost_function, algorithm=1 , d=None, max_const=500, max_var=500, alpha=0.05, scale=False):
        """ Optimize online lp covering/packing problems:

        This class declares the LP problem, updates and sattisfies constraints in an online manner
        and returns an optimal solution.

        Arguments:
        cost_function -- function that returns the cost for a specific index i (c)
        algorithm -- algorithm used to solve the sub-instance problem
        d -- max size of constraints in an specific updated (d = max_j |S(j)|)
        max_const -- maximum memory allocated for constraints (m)
        max_var -- maximum memory allocated for variables (n)
        alpha -- value set for continous updating y[j]
        scale -- (when needed) scale dual so it becomes fractional and feasible
        
        Return:
        get_primal_solution -- returns the solution for the primal objective
        get_dual_solution -- returns the solution for the dual objective

        Check:
        check_primal_feasibility -- returns a true/false flag after checking the feasibility of the solution
        """
        self.cost_function = cost_function
        self.algorithm = algorithm
        self.d = d  # must be provided for some algorithms
        if self.algorithm != 1 and not self.d:
            raise Exception("d must be provided for this algorithm")
        self.max_const = max_const
        self.max_var = max_var
        self.alpha = alpha
        self.scale = scale
        if self.scale and not self.d:
            raise Exception("d must be provided if scale is set to true")
        self.scale_factor = 1 # set up in update
        
        self.j = 0  # constraint counter (rows of S)
        self.x = np.zeros(self.max_var)
        self.y = np.zeros(self.max_const)
        self.S = np.zeros(shape=(self.max_const, self.max_var))
        
    def c(self, i):
        return self.cost_function(i)
        
    def add_x_constraint(self, j, i_indexes):
        for i in i_indexes:
            self.S[j, i] = 1
            
    def check_constraint(self, i_indexes):
        return sum(self.x[i_indexes]) < 1

    def fractional_update_primal_dual(self, i_indexes):
        self.add_x_constraint(self.j, i_indexes)
        sj_len = len(i_indexes)
        while self.check_constraint(i_indexes):
            for i in i_indexes:
                self.x[i] = self.x[i]*(1 + 1/self.c(i)) + 1/(sj_len*self.c(i))
            self.y[self.j] += 1
        self.j += 1  # increase constraint counter

    def continous_update_primal_dual(self, i_indexes):
        self.add_x_constraint(self.j, i_indexes)
        while self.check_constraint(i_indexes):
            self.y[self.j] += self.alpha
            for i in i_indexes:
                v0 = np.log(1+self.d)/self.c(i)
                i_in_sj = np.asarray(np.where(self.S[:,i]>0))[0]
                v1 = np.sum(self.y[i_in_sj])
                self.x[i] = 1/2*(np.exp(v0*v1)-1)
        self.j += 1  # increase constraint counter

    def continous_update_primal_dual2(self, i_indexes):
        self.add_x_constraint(self.j, i_indexes)
        while self.check_constraint(i_indexes):
            self.y[self.j] += self.alpha
            for i in i_indexes:
                i_in_sj = np.asarray(np.where(self.S[:,i]>0))[0]
                if self.x[i] == 0 and np.sum(self.y[i_in_sj]) == self.c(i):
                    self.x[i] = 1/self.d
            for i in i_indexes:
                if self.x[i] >= 1/self.d:
                    i_in_sj = np.asarray(np.where(self.S[:,i]>0))[0]
                    v1 = np.sum(self.y[i_in_sj])/self.c(i)
                    self.x[i] = (1/self.d)*np.exp(v1-1)
        self.j += 1

    def update_primal_dual(self, i_indexes):
        if self.algorithm == 1:
            self.fractional_update_primal_dual(i_indexes)
        elif self.algorithm == 2:
            self.continous_update_primal_dual(i_indexes)
        else:
            self.continous_update_primal_dual2(i_indexes)
        
    def get_primal_solution(self):
        nx = self.x[self.x>0]
        tc = 0
        for i,x_i in enumerate(nx):
            tc += x_i*self.c(i)
        return tc
    
    def get_dual_solution(self):
        # determine scale factor
        if self.algorithm == 1:
            self.scale_factor = np.log2(3*self.d + 1) if self.scale else 1
        elif self.algorithm == 3:
            self.scale_factor = np.log(self.d) + 1 if self.scale else 1
        # scale solution
        self.y = self.y/self.scale_factor
        # return it
        return sum(self.y)

    def check_primal_feasibility(self):
        return_flag = True
        return_flag = return_flag and np.all(self.x >= 0)  # all x >= 0
        constraints = np.unique(np.asarray(np.where(self.S>0))[0])
        for j in constraints:
            constraint_indexes = np.asarray(np.where(self.S[j,:]>0))[0]
            check = np.sum(self.x[constraint_indexes]) >= 1  # constraints greater or equal to 1
            return_flag = return_flag and check
        return return_flag

    def check_dual_feasibility(self):
        return_flag = True
        return_flag = return_flag and np.all(self.y >= 0)  # all y >= 0
        constraints = np.unique(np.asarray(np.where(self.S>0))[1])
        for i in constraints:
            constraint_indexes = np.asarray(np.where(self.S[:,i]>0))[0]
            check = np.round(np.sum(self.y[constraint_indexes]), 1) <= self.c(i)  # constraints greater or equal to c_i
            return_flag = return_flag and check
        return return_flag

    def get_S(self):
        constraints = max(np.where(self.S>0)[0])
        variables = max(np.where(self.S>0)[1])
        return self.S[:constraints, :variables]