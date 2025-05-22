#!/usr/bin/python

import vot
import sys
import time
import cv2
import numpy as np
import collections
import math

class VCTracker(object):

    def __init__(self, image, region):
        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        #Initial template
        self.template = image[int(top):int(bottom), int(left):int(right)]
        #Center position of the template (u,v)
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        #Size of the template (width, height)
        self.size = (region.width, region.height)

        self.pos = (region.x, region.y)
        self.movement_speed = np.array([])
        self.skipped_frames = 1
        
        #Use these lines for testing.
        # Comment them when you evaluate with the vot toolkit
        im = cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255,0,0), 2)
        cv2.imshow('result',im)
        cv2.imshow('template',self.template)
        cv2.waitKey(1) #change 0 to 1 - remove waiting for key press


    # *******************************************************************
    # This is the function to fill. You can also modify the class and add additional
    # helper functions and members if needed
    # It should return, in this order, the u (col) and v (row) coordinates of the top left corner
    # the width and the height of the bounding box
    # *******************************************************************
    def track(self, image):

        # Fill here the function
        # You have the information in self.template, self.position and self.size
        # You can update them and add other variables

        # Parameters
        deviation_from_mean_speed = 2.5

        # Logic

        sim_map = cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sim_map)
        
        rough_indices, values = self.get_max_indices(sim_map, 40)
        indices = self.group_indices(rough_indices, values, 40)
        closest, distance = self.get_closest_indices(indices, self.pos)
        
        is_reasonable_distance = len(self.movement_speed) == 0 or distance < self.skipped_frames * deviation_from_mean_speed * np.mean(self.movement_speed)

        if is_reasonable_distance:
            self.movement_speed = np.append(self.movement_speed, distance)
            new_pos = closest
            self.skipped_frames = 1
            acceptable_distance = 0
        else:
            new_pos = self.pos
            self.skipped_frames += 1
            acceptable_distance = self.skipped_frames * deviation_from_mean_speed * np.mean(self.movement_speed)

        #new_pos = max_loc

        # Position/template nicht updaten, wenn sehr weit weg von momentum
        # Momentum = letzte/average distance zwischen frames
        
        
        left,top = new_pos
        confidence = 1

        right = left + self.size[0]
        bottom = top + self.size[1]

        self.pos = new_pos

        if is_reasonable_distance:
            ksize = (3, 3)
            self.template = cv2.blur(image[int(top):int(bottom), int(left):int(right)], ksize)

	
        return vot.Rectangle(left, top, self.size[0], self.size[1]), confidence, indices, self.template, rough_indices, acceptable_distance
    
    def get_kernel(self, k=11, sigma=1.5):
        gaussKernel = np.zeros(shape=(k,k))

        for i in range(k):
            for j in range(k):
                gaussKernel[i,j] = np.exp(-((i-(k-1)/2)**2+(j-(k-1)/2)**2)/(2*(sigma**2)))
        
        return gaussKernel
    
    def group_indices(self, indices, values, radius):
        map = np.zeros((len(indices), len(indices)))
        for i, x in enumerate(indices):
            for j, y in enumerate(indices):
                dist = np.linalg.norm(np.array(x) - np.array(y))
                if dist < radius:
                    map[i, j] = 1

        finished = []
        groups = []
        for i in range(len(indices)):
            if i in finished:
                continue
            
            queue = [i]
            group = []

            while len(queue) > 0:
                current = queue.pop(0)
                finished.append(current)
                group.append(current)
                others = np.where(map[current] == 1)[0]
                for other in others:
                    if other not in finished and other not in queue:
                        queue.append(other)
            groups.append(group)

        new_indices = []
        for group in groups:
            max_val = 0
            representative = -1
            for member in group:
                if values[member] > max_val:
                    max_val = values[member]
                    representative = member
            new_indices.append(indices[representative])
        return new_indices

    def get_closest_indices(self, indices, pos):
        dist = math.inf
        closest = pos
        for coord in indices:
            current_pos = (coord[1], coord[0])
            current_dist = np.linalg.norm(np.array(pos) - np.array(current_pos))
            if current_dist < dist:
                dist = current_dist
                closest = current_pos
        return closest, dist
    
    def get_max_indices(self, sim_map, n):
        map = sim_map.flatten()
        max_values = np.partition(map, -n)[-n:]
        
        indices = []
        for val in max_values:
            indices.append(np.argwhere(sim_map == val)[0])

        return indices, max_values




                
        
        
        
# *****************************************
# VOT: Create VOT handle at the beginning
#      Then get the initializaton region
#      and the first image
# *****************************************
handle = vot.VOT("rectangle")
selection = handle.region()

# Process the first frame
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
image = cv2.imread(imagefile)

# Initialize the tracker
tracker = VCTracker(image, selection)

while True:
    # *****************************************
    # VOT: Call frame method to get path of the
    #      current image frame. If the result is
    #      null, the sequence is over.
    # *****************************************
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    
    # Track the object in the image  
    region, confidence, indices, template, rough_indices, acceptable_distance = tracker.track(image)

    
    #Use these lines for testing.
    # Comment them when you evaluate with the vot toolkit

    
    """ im = image.copy()
    for ind in indices:
        im = cv2.rectangle(im,(int(ind[1]),int(ind[0])),(int(ind[1]+40),int(ind[0]+42)), (0,255,0), 2)

    for ind in rough_indices:
        im = cv2.rectangle(im,(int(ind[1]),int(ind[0])),(int(ind[1]+40),int(ind[0]+42)), (0,255,0), 2)

    im = cv2.circle(im,(int(region.x),int(region.y)), int(acceptable_distance) or 0, (0,0,255), 2)
        
    im = cv2.rectangle(im,(int(region.x),int(region.y)),(int(region.x+region.width),int(region.y+region.height)), (255,0,0), 2)
    cv2.imshow('result',im)
    cv2.imshow('template',template)
    if cv2.waitKey(0) & 0xFF == ord('q'):
      break """
    
    # *****************************************
    # VOT: Report the position of the object
    #      every frame using report method.
    # *****************************************
    handle.report(region, confidence)

