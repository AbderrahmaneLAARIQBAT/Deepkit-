import numpy as np 
class LinearRegression:
    
    # Construction
    def __init__(self):
        self._theta = None 
        
    # fit function :
    def fit(self,X,y,method='normal',alpha=0.01, epochs=200,batch=10):
        
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
            
         # Validate batch
        if batch <= 0:
            raise ValueError(
                "batch must be greater than 0."
            )
        # add the intercept 
        X = np.c_[np.ones(X.shape[0]),X]
        
        # confirm the optimization method 
        if method == 'normal' :
            self._normal_equation(X,y)
            
        elif method == 'gradient':
            self._gradient_descent(X,y,alpha,epochs,batch)
            
        else : 
            raise ValueError(
                "Optimization method doesn't exist. "
                "choose 'normal' or 'gradient'.")
            
            
        return self 
    
            
        
    def predict(self, X):
        
        # check the fitting 
        if self._theta is None:
            raise ValueError("Model must be fitted before prediction.")
            
        # converts the feature matrix and target to NumPy array 
        X = np.asarray(X)
        
        # check the shape
        if X.ndim != 2 :
            raise ValueError(
                    "X must be a 2D array of shape "
                    "(n_samples, n_features)."
                )

        # add the intercept to first column X 
        X = np.c_[np.ones(X.shape[0]),X]
        
        return X @ self._theta 

    def _normal_equation(self, X, y):
        
        # uses the formula theta = inverse(X.T @ X) @ X.T @ y
        self._theta = np.linalg.solve(X.T @ X, X.T @ y)
        
    # calculate the partial Derivation 
    def _gradient(self,X,y):
        
        # calculate the prediction
        y_pred = X @ self._theta 

        # measure the error 
        error = y_pred - y
        m = len(y)
        
        # calculate the gradient 
        dtheta = (1/m)* (X.T @ error)
        
        return dtheta
        
    # batch Gradient algorithm 
    def _gradient_descent(self,X,y,alpha,epochs,batch):
        
        self._theta = np.zeros(X.shape[1])
        m = y.shape[0]
       
        for i in range(epochs):
            for start in range(0, m, batch):
                end = min(start + batch, m)
                X_batch = X[start:end]
                y_batch = y[start:end]
                dtheta = self._gradient(X_batch,y_batch)
                self._theta = self._theta - (alpha * dtheta)
        
    # measure the performance     
    @staticmethod
    def r2_score(y, y_pred):
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
    
    # return intercept
    @property
    def intercept_(self):
        return self._theta[0]
    # return coefficient
    @property
    def coef_(self):
        return self._theta[1:]
    