Before we take part in the regression algorithms, there is one more thing need to be clear.  
The **core** of the machine learning, is to find one equation  `y = f(x)`   
Just find one f then we solve the problem.  

# Linear Regression
The simplest example of f(x) is `y = mx + c`. It is the form of linear equations.  
In linear regression, we need to find a **best-fit** linear equation.  
Use the figures below, we may insert a line to fit the points inside.(left to right)
<center><img src="/lg.jpg" alt=""></center>  

What is the meaning of **best-fit** ？
**Answer:**  The predicted values will be as close as possible to 'reality'.  
De-abstraction, measure distance between predicted values and real values, which refers to a terminology 👉🏻 **residual**  
$$residual = |y - \hat{y}| = y - (mx + c)$$  
The ideal situation we expected is $y = \hat{y}$. But most of the cases, they are not. What we can do is to minimise the difference between them. This introduces to another terminology, **optimisation**. In linear regression, usually we use the squared loss as the *loss function*. I did not find a clear reason why squared loss is used. The following may be potential reasons. From a mathematical point of view, the mean square error is a convex function, which has good properties for solving parameters, and can use common optimization algorithms ( Such as gradient descent) to solve. The second reason is that it is easy to solve, and the optimal parameters can be found by taking derivatives and making the derivatives zero.  
