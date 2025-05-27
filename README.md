![Figure_1](https://github.com/user-attachments/assets/4add7223-827a-47a1-99f1-22ed0a84e7fe)

"How you structured the code"   :  
Created a class called MobiusStrip to group all the logic in one place.
Input values for radius R, width w, and resolution n
We calculate X, Y, and Z coordinates using numpy functions, which describe the Möbius strip in 3D space
Used calculus logic to find the surface area
Taken gradients and then used Simpson's rule from the scipy library to estimate the area
For each tiny segment, used the Euclidean distance (from scipy.spatial.distance) to measure the length between points and add them up
Lastly, the plot() function shows a 3D view of the strip using matplotlib

'How you approximated surface area'  :  
First, I used NumPy to calculate how the surface changes in two directions (u and v) using np.gradient.
Then, I took the cross product of these gradients to get the vectors that point perpendicular to the surface (normal vectors).
I found the length (magnitude) of each normal vector using np.linalg.norm.
This gave me tiny area pieces .
Finally, I used simpson from SciPy to add all these small areas together to get the total surface area.

"Any challenges you faced"  :  
Understanding how to calculate the surface area using vector calculus was confusing at first.
Learning to use NumPy’s gradient and cross functions correctly took some trial and error.
Figuring out how to apply Simpson’s Rule for 2D data using SciPy’s simpson function was tricky
