#!/usr/bin/python

import vot
import sys
import time
import cv2
import numpy as np
import collections
import math

debug_mode = False

class VCTracker(object):

    def __init__(self, image, region):
        # Parameters
        self.deviation_from_mean_speed = 3
        self.n_max_sim_values = 30
        self.current_to_orignal_template_tradeoff = 0.5
        self.scales = [1]
        self.rotations = [-10, -5, 0, 5, 10]
        self.speed_decay = 0.9

        self.window = max(region.width, region.height) * 2
        #Original Size of the object (width, height)
        self.size = (int(region.width), int(region.height))
        #Center position of the template (u,v)
        self.position = (region.x + region.width / 2, region.y + region.height / 2)

        if self.size[0] > self.size[1]:
            self.template_size = self.size[0]
        else:
            self.template_size = self.size[1]

        left = int(max(region.x - ((self.template_size - self.size[0]) // 2), 0))
        top = int(max(region.y - ((self.template_size - self.size[1]) // 2), 0))

        right = min(left + self.template_size, image.shape[1] - 1)
        bottom = min(top + self.template_size, image.shape[0] - 1)

        #Initial template
        self.set_template(cv2.blur(image[int(top):int(bottom), int(left):int(right)], (3, 3), cv2.BORDER_DEFAULT))

        self.original_template = self.template.copy()
        self.pos = (left, top)
        self.last_move = np.array([0, 0])
        self.movement_speed = np.array([])
        self.skipped_frames = 1
        self.rotation = 0
        # Update template size and use that instead?
        self.scale = 1
        _, binary = cv2.threshold(cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        self.template_mask = binary

        self.current_report = self.TrackReport(region, 1, None)
        self.acceptable_distance = 0


        
        #Use these lines for testing.
        # Comment them when you evaluate with the vot toolkit
        if debug_mode:
            im = cv2.rectangle(image, (int(region.x), int(region.y)), (int(region.x+self.size[0]), int(region.y+self.size[1])), (255,0,0), 2)
            cv2.imshow('result',im)
            cv2.imshow('template',self.template)
            cv2.waitKey(0) #change 0 to 1 - remove waiting for key press

    class TrackResult(object):
        def __init__(self, position, template_size, confidence, scale, rotation, distance, is_reasonable_distance, mask=None):
            self.position = position
            self.template_size = template_size
            self.confidence = confidence
            self.scale = scale
            self.rotation = rotation
            self.mask = mask
            self.distance = distance
            self.is_reasonable_distance = is_reasonable_distance

        def __repr__(self):
            return f"""{{
                {self.position},
                {self.confidence},
                {self.distance},
                {self.is_reasonable_distance},
            }}"""

    class TrackReport(object):
        def __init__(self, rect, confidence, misc):
            self.rect = rect
            self.confidence = confidence
            self.misc = misc
            


    def template_to_object(self, point):
        return (int(point[0] + ((self.template_size - self.size[0]) // 2)), int(point[1] + ((self.template_size - self.size[1]) // 2)))
    
    def object_to_template(self, point):
        return (int(point[0] - ((self.template_size - self.size[0]) // 2)), int(point[1] - ((self.template_size - self.size[1]) // 2)))
    
    def set_template(self, patch, mask=None):
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            return
        self.template = patch.copy()
        if mask is not None:
            self.template_mask = mask.copy()
        else:
            self.template_mask = np.ones((patch.shape[0], patch.shape[1]), dtype=np.uint8) * 255


    # *******************************************************************
    # This is the function to fill. You can also modify the class and add additional
    # helper functions and members if needed
    # It should return, in this order, the u (col) and v (row) coordinates of the top left corner
    # the width and the height of the bounding box
    # *******************************************************************
    def track(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = []

        for rotation in self.rotations:
            for scale in self.scales:
                results.append(self.execute_track(image_gray, rotation, scale))

        # Select the best candidate based on confidence & reasonable distance
        best_result = None
        for result, _ in results:
            if result.is_reasonable_distance and (best_result is None or result.confidence > best_result.confidence):
                best_result = result

        if best_result:
            # Update position, speed, and template
            self.last_move = np.array(best_result.position) - np.array(self.pos)
            self.pos = best_result.position
            self.movement_speed = self.speed_decay * self.movement_speed + (1 - self.speed_decay) * best_result.distance
            self.scale *= best_result.scale
            self.rotation += best_result.rotation
            self.set_template(image[int(self.pos[1]):int(self.pos[1] + best_result.template_size),
                                    int(self.pos[0]):int(self.pos[0] + best_result.template_size)],
                            best_result.mask)
            x, y = map(int, self.pos)
            coords = cv2.findNonZero(best_result.mask)
            left, top, width, height = cv2.boundingRect(coords)
            self.previous_result = best_result
            return vot.Rectangle(x + left, y + top, width, height), best_result.confidence, None, (self.acceptable_distance, [], [])

        else:
            # Fallback: predict based on last motion
            self.pos = (self.pos[0] + self.last_move[0], self.pos[1] + self.last_move[1])
            return vot.Rectangle(self.pos[0], self.pos[1], self.size[0], self.size[1]), 0.5, None, (self.acceptable_distance, [], [])

    

    def transform(self, image, rotation, size):

        image = cv2.resize(image, (size, size))
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, rotation, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def execute_track(self, image_gray, rotation, scale):
        current_size = int(self.template_size * self.scale * scale)
        t_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        t_resized = self.transform(t_gray, rotation + self.rotation, current_size)
        mask_resized = self.transform(self.template_mask, rotation + self.rotation, current_size)

        sim_map = cv2.matchTemplate(image_gray, t_resized, cv2.TM_CCOEFF_NORMED, mask=mask_resized)
        sim_map = np.nan_to_num(sim_map)

        max_loc = cv2.minMaxLoc(sim_map)[3]
        confidence = sim_map[max_loc[1], max_loc[0]]
        position = (max_loc[0], max_loc[1])
        distance = np.linalg.norm(np.array(position) - np.array(self.pos))
        is_reasonable = self.acceptable_distance == 0 or distance < self.acceptable_distance

        return self.TrackResult(position, current_size, confidence, scale, rotation, distance, is_reasonable, mask_resized), None


    
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
    region, confidence, _, (acceptable_distance, indices, rough_indices) = tracker.track(image)

    
    #Use these lines for testing.
    # Comment them when you evaluate with the vot toolkit
    if debug_mode:
        #contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        #mask_size = mask.shape[0]
        
        im = image.copy()
        for ind in rough_indices:
            im = cv2.rectangle(im,(int(ind[1]),int(ind[0])),(int(ind[1]+region.width),int(ind[0]+region.height)), (0,255,0), 2)

        for ind in indices:
            im = cv2.rectangle(im,(int(ind[1]),int(ind[0])),(int(ind[1]+region.width),int(ind[0]+region.height)), (0,100,255), 2)

        im = cv2.circle(im,(int(region.x),int(region.y)), int(acceptable_distance) or 0, (0,0,255), 2)
            
        im = cv2.rectangle(im,(int(region.x),int(region.y)),(int(region.x+region.width),int(region.y+region.height)), (255,0,0), 2)
        
        cv2.imshow('result',im)
        #cv2.imshow('template',template)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break   
    
    # *****************************************
    # VOT: Report the position of the object
    #      every frame using report method.
    # *****************************************
    handle.report(region, confidence)

