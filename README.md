# Optical Flow and object tracker using Lucas-Kanade Algorithm

The Lucas-Kanade method is used for optical flow estimation, to track desired features in a video.
The project aims to implement the Lucas-Kanade (LK) template tracker.
- The LK tracker that is implemented [here](./Code/lucaskanade.py) is used to track:
    - A car moving on a road,
    - Face of a baby fighting with a dragon,
    - Usain Bolt running on a track, respectively in three video sequences.

### Approach and Implementation:
- To initialize the tracker a template is defined by drawing a bounding box around the object to be tracked in the first frame of the video. 
- For each of the subsequent frames the tracker updates an affine transform that warps the current frame so that the template in the first frame is aligned with the warped
current frame.
- At the core of the algorithm is a function **_affineLKtracker_(img, tmp, rect, p<sub>prev</sub>)**.
- This function gets as input a grayscale image of the current frame (**img**), the template image
(**tmp**), the bounding box (**rect**) that marks the template region in **tmp**, and the parameters p<sub>prev</sub> of the previous warping.
- The function iteratively (gradient descent) computes the new warping parameters p<sub>new</sub>, and returns these
parameters.
- The algorithm computes the affine transformations from the template to every frame in the sequence and draws the bounding boxes of the rectangles warped from the first frame.
- Elaborate explanation about the approach and the pipeline can be found in the [report](Report.pdf)

### Output:

**Tracking moving car (video [here](https://drive.google.com/file/d/1UAPNs9cprUpfJuVWVzzGFTA9Ix17ecIG/view?usp=sharing)):**

![alt text](./output/track_car.PNG?raw=true "Tracking moving car")


**Tracking baby face (video [here](https://drive.google.com/file/d/1sC5zJaDpZaEOKO5GB0fOIiAbLdJbqW6I/view?usp=sharing)):**

![alt text](./output/track_baby.PNG?raw=true "Tracking baby face")


**Tracking running Bolt (video [here](https://drive.google.com/file/d/1RU3QxBeAduXsoll0UXBcejsySumBOgBT/view?usp=sharing)):**

![alt text](./output/track_bolt.PNG?raw=true "Tracking running Bolt")



### Instructions to run the code:

Input dataset required for the code is present in:
- [Car dataset](./Code/Car/img)
- [Baby dataset](./Code/DragonBaby/img)
- [Bolt dataset](./Code/Bolt/img)

Go to directory:  _cd 'Code/_
- To run the car tracker run: 
    - $ _python trackCar.py_ 

- To run the baby face tracker run: 
    - $ _python trackBaby.py_

- To run the Bolt tracker run: 
    - $ _python trackBolt.py_ 



### References:
- [Simon and Matthew’s paper - Lucas-Kanade 20 Years On: A Unifying Framework](https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf)
- [Lecture on Lucas-Kanade Tracker (KLT) by Dr. Mubarak Shah (University of Florida)](https://www.youtube.com/watch?v=tzO245uWQxA)
- [Optical Flow with Lucas-Kanade method - OpenCV 3.4 with Python 3](https://www.youtube.com/watch?v=7soIa95QNDk)
- [Optical-Flow using Lucas Kanade for Motion Tracking](https://www.youtube.com/watch?v=1r8E9uAcn4E)
