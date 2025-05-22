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
        #Original Size of the object (width, height)
        self.size = (int(region.width), int(region.height))
        #Center position of the template (u,v)
        self.position = (region.x + region.width / 2, region.y + region.height / 2)

        if self.size[0] > self.size[1]:
            self.template_size = (self.size[0], self.size[0])
        else:
            self.template_size = (self.size[1], self.size[1])

        left = int(max(region.x - ((self.template_size[0] - self.size[0]) // 2), 0))
        top = int(max(region.y - ((self.template_size[1] - self.size[1]) // 2), 0))

        right = min(left + self.template_size[0], image.shape[1] - 1)
        bottom = min(top + self.template_size[1], image.shape[0] - 1)

        #Initial template
        self.set_template(cv2.blur(image[int(top):int(bottom), int(left):int(right)], (3, 3), cv2.BORDER_DEFAULT))
        

        self.original_template = self.template.copy()
        self.pos = (left, top)
        self.movement_speed = np.array([])
        self.skipped_frames = 1
        
        #Use these lines for testing.
        # Comment them when you evaluate with the vot toolkit
        im = cv2.rectangle(image, (int(region.x), int(region.y)), (int(region.x+self.size[0]), int(region.y+self.size[1])), (255,0,0), 2)
        cv2.imshow('result',im)
        cv2.imshow('template',self.template)
        cv2.waitKey(0) #change 0 to 1 - remove waiting for key press

    def template_to_object(self, point):
        return (int(point[0] + ((self.template_size[0] - self.size[0]) // 2)), int(point[1] + ((self.template_size[1] - self.size[1]) // 2)))
    
    def object_to_template(self, point):
        return (int(point[0] - ((self.template_size[0] - self.size[0]) // 2)), int(point[1] - ((self.template_size[1] - self.size[1]) // 2)))
    
    def set_template(self, template):
        self.template = template
        self.template[:, 0:(self.template_size[0] - self.size[0]) // 2,:] = 0
        self.template[:, (self.template_size[0] - self.size[0]) // 2 + self.size[0]:,:] = 0
        self.template[0:(self.template_size[1] - self.size[1]) // 2,:,:] = 0
        self.template[(self.template_size[1] - self.size[1]) // 2 + self.size[1]:,:,:] = 0


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

        # Square template to the bigger dimension
        # Rotate template 8 times and get best fit
        # report rotation + original size

        # Parameters
        deviation_from_mean_speed = 3
        n_max_sim_values = 30

        # Logic
        # Save original template for comparing/averaging
        template = (0.5 * self.template + 0.5 * self.original_template).astype('uint8')

        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)


        #cv2.imshow('bin', binary)
        #cv2.waitKey(0)

        blur_image = cv2.blur(image, (3, 3), cv2.BORDER_DEFAULT)
        #gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)

        sim_map = cv2.matchTemplate(blur_image, template, cv2.TM_CCOEFF_NORMED, mask=binary)
        sim_map = np.nan_to_num(sim_map, nan=0.0)

        #cv2.imshow('simmap', sim_map)
        #cv2.waitKey(0)

        # Get top n similarity indices
        rough_indices, values = self.get_max_indices(sim_map, n_max_sim_values)

        # Group indices and get a representative for each
        indices, values = self.group_indices(rough_indices, values, self.template_size[0])

        # Get closest to last frame
        closest, distance, confidence = self.get_closest_indices(indices, values, self.pos)
        
        # Check if travel distance seems reasonable given the previous observations
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
        
        
        left,top = new_pos

        right = left + self.template_size[0]
        bottom = top + self.template_size[1]

        self.pos = new_pos

        if is_reasonable_distance:
            self.set_template(cv2.blur(image[int(top):int(bottom), int(left):int(right)], (3, 3), cv2.BORDER_DEFAULT))

	
        left, top = self.template_to_object((left, top))

        return vot.Rectangle(left, top, self.size[0], self.size[1]), confidence, indices, self.template, rough_indices, acceptable_distance, self.size, self.template_size
    
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
        new_values = []
        for group in groups:
            max_val = 0
            representative = -1
            for member in group:
                if values[member] > max_val:
                    max_val = values[member]
                    representative = member
            new_indices.append(indices[representative])
            new_values.append(max_val)
        return new_indices, new_values

    def get_closest_indices(self, indices, values, pos):
        dist = math.inf
        closest = pos
        closest_value = 0
        for coord, value in zip(indices, values):
            current_pos = (coord[1], coord[0])
            current_dist = np.linalg.norm(np.array(pos) - np.array(current_pos))
            if current_dist < dist:
                dist = current_dist
                closest = current_pos
                closest_value = value
        return closest, dist, closest_value
    
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
    region, confidence, indices, template, rough_indices, acceptable_distance, size, template_size = tracker.track(image)

    
    #Use these lines for testing.
    # Comment them when you evaluate with the vot toolkit

    
    im = image.copy()
    for ind in rough_indices:
        im = cv2.rectangle(im,(int(ind[1]),int(ind[0])),(int(ind[1]+template_size[0]),int(ind[0]+template_size[1])), (0,255,0), 2)

    for ind in indices:
        im = cv2.rectangle(im,(int(ind[1]),int(ind[0])),(int(ind[1]+template_size[0]),int(ind[0]+template_size[1])), (0,100,255), 2)

    im = cv2.circle(im,(int(region.x),int(region.y)), int(acceptable_distance) or 0, (0,0,255), 2)
        
    im = cv2.rectangle(im,(int(region.x),int(region.y)),(int(region.x+region.width),int(region.y+region.height)), (255,0,0), 2)
    cv2.imshow('result',im)
    cv2.imshow('template',template)
    if cv2.waitKey(0) & 0xFF == ord('q'):
      break
    
    # *****************************************
    # VOT: Report the position of the object
    #      every frame using report method.
    # *****************************************
    handle.report(region, confidence)

