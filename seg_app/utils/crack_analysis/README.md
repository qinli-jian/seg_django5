# A-Hybrid-Method-for-Pavement-Crack-Width-Measurement
A method to calculate the width of a binary image as described in: https://www.sciencedirect.com/science/article/abs/pii/S026322412200505X#:~:text=The%20hybrid%20method%20obtains%20the,close%20to%20the%20orthogonal%20direction.

The method identifies a group of points that are close to the orthogonal projection vector and selects a pair of points that is the shortest as shown in the figure below. The green lines are general measurements while the red line indicates the maximum width. <br>

![comparison of methods](/comparison_measurement.png) <br>

The benefits of the hybrid method are as follows:
   <ul>
     <li>Allows the user to tune the strength of the orthogonal projection and the shortest method</li>
     <li>Prevents overestimation effects by the orthogonal projection method when cracks are curved</li>
     <li>More distinct crack width measurements than the shortest method</li>
     <li>More robust to binary images that have unparallel boundaries</li>
   </ul>
   
How to use:
   <ol>
    <li>Input binary image</li>
    <li>Tune the pruning strength</li>
    <li>Tune the balancing coefficient</li>
   </ol>


   
