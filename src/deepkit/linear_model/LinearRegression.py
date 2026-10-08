import numpy as np 



class LinearRegression:
    
    # Construction
    def __init__(self, solver='normal',alpha=0.01,epochs=200, batch_size=32):
        # Parameters 
        self._theta = None 
        self._solver = solver
        self._alpha =  alpha
        if self._alpha <= 0:
            raise ValueError("alpha must be greater than 0.")  
        self._epochs = epochs
        if self._epochs <= 0:
            raise ValueError("epochs must be greater than 0.")
        self._batch_size = batch_size 
        # Validate batch
        if self._batch_size <= 0:
            raise ValueError(
                "batch must be greater than 0."
            )
        # Model information 
        self.n_features_in_ = None 
        self.n_iter_ = None 
        # Training history
        self.cost_history_ = []

    # =============================
    #      Validate the Input    
    # =============================
    
    @staticmethod
    def _validate_input(X,y):
        
        # converts the feature matrix and target to NumPy array 
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        
        # X must be 2D
        if X.ndim != 2 :
            raise ValueError(
                "X must be a 2D array of shape "
                "(n_samples, n_features)."
            )
            
         # Same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )
            
         
        return X,y
        
    # ===============================================
    # CHECK FITTED 
    # ===============================================

    def _check_fitted(self):
        if self._theta is None:
            raise ValueError(
                "Model must be fitted before"
                "accessing its parameters."
            )

    # ==================================================
    # FIT FUNCTION 
    # ==================================================
    
    def fit(self,X,y):
        
        # Validate and convert inpu 
        X,y = self._validate_input(X,y)

        # Store number of features before adding incept 
        self.n_features_in_ =  X.shape[1]
        
        # Add the intercept column 
        X = self._add_intercept(X)

        # reset Training history 
        self.cost_history_= []
        
        # confirm the optimization method 
        if self._solver == 'normal' :
            self._normal_equation(X,y)
            
        elif self._solver == 'batch':
            self._batch_gradient_descent(X,y)
            
        elif self._solver == 'stochastic':
            self._stochastic_gradient_descent(X,y)

        elif self._solver == 'mini-batch':
            self._mini_gradient_descent(X,y)
            
        else : 
            raise ValueError(
                "Optimization method doesn't exist. "
                "choose 'normal', 'batch', 'stochastic', 'mini-batch' .")
            
            
        return self 
    
    # ===============================
    # INITIALIZE THETA 
    # ===============================
    def _initialize_theta(self, X):
        self._theta = np.zeros(X.shape[1])

    # ===============================
    # ADD INTERCEPT 
    #================================
    
    @staticmethod
    def _add_intercept(X):
        return np.c_[np.ones(X.shape[0]),X]
        

    # ================================
    # PREDICT FUNCTION 
    # ================================
        
    def predict(self, X):
        
        # check the fitting 
        self._check_fitted()
            
        # converts the feature matrix and target to NumPy array 
        X = np.asarray(X)
        
        # check the shape must be 2D
        if X.ndim != 2 :
            raise ValueError(
                    "X must be a 2D array of shape "
                    "(n_samples, n_features)."
                )
            
        # Check number of features
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must contain {self.n_features_in_} features, "
                f"but received {X.shape[1]}."
            )
            
        # add the intercept to first column X 
        X = np.c_[np.ones(X.shape[0]),X]
        
        return X @ self._theta 

    
    # ==================================
    # check numerical stability 
    # ==================================
    def _check_numerical_stability(self, value):
        if not np.all(np.isfinite(value)):
            raise ValueError(
                "Training diverged due to numerical instability. "
                "Try reducing alpha or scaling your features."
            )
        
    # ===================================
    # Calculate the Gradient 
    # ===================================
    
    def _gradient(self,X,y):
        
        # calculate the prediction
        y_pred = X @ self._theta 

        # measure the error 
        error = y_pred - y
        
        m = len(y)
        
        # calculate the gradient 
        dtheta = (1/m)* (X.T @ error)
        
        return dtheta

    # ===========================================
    # COST FUNCTION 
    # ===========================================
    def _cost(self,X,y):
        
        y_pred = X @ self._theta 
        error = y_pred - y
        m = len(y)

        return (1 / (2 * m)) * np.sum(error ** 2)


    # ==================================
    # NORMAL EQUATION 
    # ==================================

    def _normal_equation(self, X, y):
        
        # uses the formula theta = (X.T X)^(-1) X.T y
        # I won't use : np.linalg.solve(X.T @ X, X.T @ y)
        # I used pinv to prevent singular matrix error when X isn't invertible 
        self._theta = np.linalg.pinv(X) @ y
        self.n_iter_ = 1

        
    # ========================================
    # BATCH GRADDIENT DESCENT 
    # =========================================
    def _batch_gradient_descent(self, X, y):
        self._initialize_theta(X)
        
        for epochs in range(self._epochs):
            
            dtheta =  self._gradient(X,y)
            self._theta -= self._alpha * dtheta

            self._check_numerical_stability(self._theta)

            cost = self._cost(X,y)
            self.cost_history_.append(cost)

        self.n_iter_ = self._epochs
             
    
    # ===================================================
    # STOCHASTIC GRADDEINT DESCENT 
    # ===================================================
    def _stochastic_gradient_descent(self, X,y):
        self._initialize_theta(X)
        m = len(y)
        for epochs in range(self._epochs):
            for i in range(m):
                
                X_i = X[i:i +1]
                y_i = y[i: i+1]

                dtheta = self._gradient(X_i, y_i)
                self._theta -= self._alpha * dtheta

            self._check_numerical_stability(self._theta)
            
            cost = self._cost(X,y)
            self.cost_history_.append(cost)

        self.n_iter_ = self._epochs

    # ============================================
    # MINI BATCH GRADIENT ALGORITHM
    # ============================================
    
    def _mini_gradient_descent(self,X,y):
        
        self._initialize_theta(X)
        m = y.shape[0]
       
        for i in range(self._epochs):
            for start in range(0, m, self._batch_size):
                end = min(start + self._batch_size, m)
                X_batch = X[start:end]
                y_batch = y[start:end]
                dtheta = self._gradient(X_batch,y_batch)
                self._theta -= self._alpha * dtheta
            cost =  self._cost(X,y)
            self.cost_history_.append(cost)
            
        self.n_iter_ = self._epochs
    

                 
    # =======================================
    # MEASURE THE MODEL PERFORMANCE 
    # =======================================
    
    @staticmethod
    def r2_score(y, y_pred):
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
        
    # ========================================
    # INTERCEPT  AND COEFFICIENT 
    # ========================================
    
    @property
    def intercept_(self):
        self._check_fitted()
        return self._theta[0]

    
    @property
    def coef_(self):
        self._check_fitted()
        return self._theta[1:]

    # ==================================
    # PARAMETERS
    # ==================================

    @property
    def theta_(self):
        self._check_fitted()
        return self._theta.copy()
    