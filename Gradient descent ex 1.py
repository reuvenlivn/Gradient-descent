
#### Gradient Descent #####
import numpy as np
#1
#x and y
x=np.array([0,1,2],dtype=np.float32)
Y=np.array([1,3,7],dtype=np.float32)
X=np.c_[np.ones_like(x),x,x**2]

start = np.array([2,2,0],dtype=np.float32)
   
def my_loss(res,y):
#   print('res=',res, '\ty:',y)
    loss = 0.5*((res-y)**2).mean()
#    print('my_loss:',loss)
    return(loss)
    
def my_loss_grad(res,y):
     return (np.dot(res-y,X))/len(X)
 
def my_model(t):
    return np.dot(X,t)

def gradient_descent(Y, start, rate, iterations):
    location=start.copy()  
    for iter in range(iterations):
        res=my_model(location)
        loss=my_loss(res,Y)
        grad=my_loss_grad(res,Y)
        location -= rate*grad
        print('iter\t {}, loss {}, new {}'.format(iter,loss,location))
 

def momentum_gradient_descent(Y, start, rate, momentum_decay, iters):
    location=start.copy()
    step=np.zeros_like(start)
#    print("step\n",step)
    for iter in range(iters):
        res=my_model(location)
        loss=my_loss(res,Y)
        grad=my_loss_grad(res,Y)
        step=momentum_decay*step - rate*grad
#        print("step\n",step)
        location+= step
        print('iter {}, loss {}, new {}'.format(iter,loss,location))


def nesterov_gradient_descent(Y, start, rate, momentum_decay, iters):
    location=start.copy()
    step=np.zeros_like(start)
    for iter in range(iters):
        res= my_model(location)
        loss=my_loss(res,Y)
        grad=my_loss_grad(np.dot(X, location+momentum_decay*step),Y)
        step= momentum_decay*step - rate*grad
        location += step
        print('iter {}, loss {}, new {}'.format(iter,loss,location))
              
#1.1
learning_rate =0.3
iters = 50
print('\n 1.1 gradient_descent iters {}, learning_rate {}'.format(iters,learning_rate))
gradient_descent(Y, start, learning_rate, iters)

learning_rate = 0.2
print('\n 1.1 gradient_descent iters {}, learning_rate {}'.format(iters,learning_rate))
gradient_descent(Y, start, learning_rate, iters)

#learning_rate = 0.01
#print('\n 1.1 gradient_descent iters {}, learning_rate {}'.format(iters,learning_rate))
#gradient_descent(Y, start, learning_rate, iters)
##1.3
#learning_rate = 0.085
#print('\n 1.3 momentum_gradient_descent iters {}, learning_rate {}'.format(iters,learning_rate))
#momentum_gradient_descent(Y, start, learning_rate, 0.9, iters)
#
##1.4
#learning_rate = 0.1
#print('\n 1.4 nesterov_gradient_descent iters {}, learning_rate {}'.format(iters,learning_rate))
#nesterov_gradient_descent(Y, start, learning_rate, 0.9, iters)
