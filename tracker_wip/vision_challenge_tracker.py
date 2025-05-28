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


        # Parameters
        self.deviation_from_mean_speed = 3
        self.n_max_sim_values = 20
        
        #Use these lines for testing.
        # Comment them when you evaluate with the vot toolkit
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
    
    def set_template(self, template, mask=None):
        self.template = template.copy()

        if mask is None:
            self.template[:, 0:(self.template_size - self.size[0]) // 2,:] = 0
            self.template[:, (self.template_size - self.size[0]) // 2 + self.size[0]:,:] = 0
            self.template[0:(self.template_size - self.size[1]) // 2,:,:] = 0
            self.template[(self.template_size - self.size[1]) // 2 + self.size[1]:,:,:] = 0
        else:
            self.template = cv2.bitwise_and(self.template, self.template, mask = mask)


    # *******************************************************************
    # This is the function to fill. You can also modify the class and add additional
    # helper functions and members if needed
    # It should return, in this order, the u (col) and v (row) coordinates of the top left corner
    # the width and the height of the bounding box
    # *******************************************************************
    def track(self, image):
        scales = [1]
        rotations = [0, -20, 20]

        # Check if travel distance seems reasonable given the previous observations
        if len(self.movement_speed > 0):
            self.acceptable_distance = self.skipped_frames * self.deviation_from_mean_speed * np.mean(self.movement_speed)
        else:
            self.acceptable_distance = 0

        results = []
        for rotation in rotations:
            for scale in scales:
                results.append(self.execute_track(image, rotation, scale))

        track_result, misc = None, None
        indices = []
        rough_indices = []
        for (t_result, m) in results:
            indices += m[1]
            rough_indices += m[3]
            if (t_result.is_reasonable_distance and # If distance is reasonable and
                ((track_result is None) or # If not assigned yet or
                (track_result is not None and t_result.confidence > track_result.confidence))): # If confidence is larger 
                track_result = t_result
                misc = m

        if track_result is not None:
            print(track_result.is_reasonable_distance, self.acceptable_distance, track_result.distance)

        if track_result is not None:
            new_pos = track_result.position
            confidence = track_result.confidence

            self.movement_speed = np.append(self.movement_speed, track_result.distance)
            self.last_move = np.array(new_pos) - np.array(self.pos)
            self.skipped_frames = 1

            self.rotation += track_result.rotation
            self.scale *= track_result.scale

        else:
            new_pos = (self.current_report.rect.x + self.last_move[0], self.current_report.rect.y + self.last_move[1])
            self.skipped_frames += 1
            confidence = self.current_report.confidence
        
        
        left,top = new_pos
        self.pos = new_pos

        

        if track_result is not None:
            right = left + track_result.template_size
            bottom = top + track_result.template_size

            self.set_template(cv2.blur(image[int(top):int(bottom), int(left):int(right)], (3, 3), cv2.BORDER_DEFAULT), track_result.mask)

	
            # Report region based on binary mask bounding box
            coords = cv2.findNonZero(track_result.mask)
            left, top, width, height = cv2.boundingRect(coords)
            left = left + new_pos[0]
            top = top + new_pos[1]

            self.current_report = self.TrackReport(vot.Rectangle(left, top, width, height), confidence, misc)

        else:
            left, top = new_pos
            width, height = self.current_report.rect.width, self.current_report.rect.height
            self.current_report = self.TrackReport(vot.Rectangle(left, top, width, height), self.current_report.confidence, self.current_report.misc)

        return self.current_report.rect, self.current_report.confidence, self.current_report.misc, (self.acceptable_distance, indices, rough_indices)
    
    

    def transform(self, image, rotation, size):

        image = cv2.resize(image, (size, size))
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, rotation, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def execute_track(self, image, rotation, scale):
        # Fill here the function
        # You have the information in self.template, self.position and self.size
        # You can update them and add other variables

        # Use binary template to set template at the end for new template shape


        # Logic
        # Save original template for comparing/averaging

        current_template_size = int(self.template_size * self.scale * scale)
        transformed_template = self.transform(self.template, rotation, current_template_size)
        transformed_original_template = self.transform(self.original_template, rotation + self.rotation, current_template_size)
        binary_template = self.transform(self.template_mask, self.rotation + rotation, current_template_size)

        
        template = (0.3 * transformed_template + 0.7 * transformed_original_template).astype('uint8')

        cv2.imshow('working template', template)
        #cv2.waitKey(0)


        #cv2.imshow('bin', binary_template)
        #cv2.waitKey(0)

        blur_image = cv2.blur(image, (3, 3), cv2.BORDER_DEFAULT)
        #gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)

        sim_map = cv2.matchTemplate(blur_image, template, cv2.TM_CCOEFF_NORMED, mask=binary_template)
        sim_map = np.nan_to_num(sim_map, nan=0.0)

        #cv2.imshow('simmap', sim_map)
        #cv2.waitKey(0)

        # Get top n similarity indices
        rough_indices, values = self.get_max_indices(sim_map, self.n_max_sim_values)

        # Group indices and get a representative for each
        indices, values = self.group_indices(rough_indices, values, self.template_size)

        # Get closest to last frame
        closest, distance, confidence = self.get_closest_indices(indices, values, self.pos)
        
        # Check if travel distance seems reasonable given the previous observations
        is_reasonable_distance = self.acceptable_distance == 0 or distance < self.acceptable_distance

        return self.TrackResult(closest, current_template_size, confidence, scale, rotation, distance, is_reasonable_distance, binary_template), (binary_template, indices, self.template, rough_indices, self.acceptable_distance, self.size, self.template_size)
    
    

    
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

def get_contour(mask):
    cv2.imshow('mask', mask)
    
    return contours

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
    region, confidence, (mask, _, template, _, _, _, _), (acceptable_distance, indices, rough_indices) = tracker.track(image)

    
    #Use these lines for testing.
    # Comment them when you evaluate with the vot toolkit
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    mask_size = mask.shape[0]
    
    im = image.copy()
    for ind in rough_indices:
        im = cv2.rectangle(im,(int(ind[1]),int(ind[0])),(int(ind[1]+mask_size),int(ind[0]+mask_size)), (0,255,0), 2)

    for ind in indices:
        im = cv2.rectangle(im,(int(ind[1]),int(ind[0])),(int(ind[1]+mask_size),int(ind[0]+mask_size)), (0,100,255), 2)

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

