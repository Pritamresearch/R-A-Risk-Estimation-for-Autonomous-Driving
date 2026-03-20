class DepthSmoother:

    def __init__(self,beta=0.7):

        self.beta = beta
        self.prev = None

    def update(self,depth):

        if self.prev is None:

            self.prev = depth

        else:

            self.prev = self.beta*depth + (1-self.beta)*self.prev

        return self.prev