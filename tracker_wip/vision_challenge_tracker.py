#!/usr/bin/python

import vot
import sys
import cv2
import numpy as np
import math
from transform_track_result import TransformTrackResult
from track_report import TrackReport

# Set to True to show images and progress manually
debug_mode = True

class VCTracker(object):

    def __init__(self, image, region):
        ### Parameters ###
        self.deviation_from_mean_speed = 3
        self.n_max_sim_values = 30
        self.current_to_orignal_template_tradeoff = 0.5
        self.scales = [1]
        self.rotations = [0, -1, 1]

        ### Sizes and Bounds ###
        #Original Size of the object (width, height)
        self.size = (int(region.width), int(region.height))

        # Set template size to the bigger of width and height
        self.template_size = max(self.size[0], self.size[1])

        left = int(max(region.x - ((self.template_size - self.size[0]) // 2), 0))
        top = int(max(region.y - ((self.template_size - self.size[1]) // 2), 0))

        right = min(left + self.template_size, image.shape[1] - 1)
        bottom = min(top + self.template_size, image.shape[0] - 1)

        ### Template Initialization ###
        #Initial template
        initial_template = self.trim_template(cv2.blur(image[int(top):int(bottom), int(left):int(right)], (3, 3), cv2.BORDER_DEFAULT))
        self.template = initial_template
        self.template_position = (left, top)

        # Template mask shape
        _, binary = cv2.threshold(cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        self.template_mask = binary

        # Save original template
        self.original_template = self.template.copy()


        ### Persistent Variables ###
        # Variables to track movement of the object
        self.last_move = np.array([0, 0])
        self.movement_speed = np.array([])
        self.skipped_frames = 1
        self.acceptable_distance = 0

        # Current Transformation of the Template
        self.rotation = 0
        self.scale = 1
        
        # Initial Report
        self.current_report = TrackReport(region, 1, None)

        #Use these lines for testing.
        # Comment them when you evaluate with the vot toolkit
        if debug_mode:
            im = cv2.rectangle(image, (int(region.x), int(region.y)), (int(region.x+self.size[0]), int(region.y+self.size[1])), (255,0,0), 2)
            cv2.imshow('result',im)
            cv2.imshow('template',self.template)
            cv2.waitKey(0) #change 0 to 1 - remove waiting for key press


    def trim_template(self, template):
        result = template.copy()
        result[:, 0:(self.template_size - self.size[0]) // 2,:] = 0
        result[:, (self.template_size - self.size[0]) // 2 + self.size[0]:,:] = 0
        result[0:(self.template_size - self.size[1]) // 2,:,:] = 0
        result[(self.template_size - self.size[1]) // 2 + self.size[1]:,:,:] = 0
        return result

    def set_masked_template(self, template, mask):
        self.template = template.copy()
        self.template = cv2.bitwise_and(self.template, self.template, mask = mask)


    # *******************************************************************
    # This is the function to fill. You can also modify the class and add additional
    # helper functions and members if needed
    # It should return, in this order, the u (col) and v (row) coordinates of the top left corner
    # the width and the height of the bounding box
    # *******************************************************************
    def track(self, image):
        # Set the reasonable distance for object movement
        if len(self.movement_speed > 0):
            self.acceptable_distance = self.skipped_frames * self.deviation_from_mean_speed * np.mean(self.movement_speed)
        else:
            self.acceptable_distance = 0

        # Apply all transformations and collect results
        results = []
        for rotation in self.rotations:
            for scale in self.scales:
                results.append(self.execute_track(image, rotation, scale))

        # Get a reasonable track result from all attempts
        track_result, misc = None, None
        indices = []
        rough_indices = []
        for (t_result, m) in results:
            indices += m[2]
            rough_indices += m[3]
            if (t_result.is_reasonable_distance and # If distance is reasonable and
                ((track_result is None) or # If not assigned yet or
                (track_result is not None and t_result.confidence > track_result.confidence))): # If confidence is larger 
                track_result = t_result
                misc = m

        # Create new Track Report depending on if a suitable track result was found
        if track_result is not None:
            new_report = self.handle_track_result(track_result, misc)
        else:
            new_report = self.handle_no_result()

        self.current_report = new_report
        return new_report.rect, new_report.confidence, new_report.misc, (self.acceptable_distance, indices, rough_indices)
    
    def handle_no_result(self):
        # New position is the current position + the last movement of the object
        new_position = (self.current_report.rect.x + self.last_move[0], self.current_report.rect.y + self.last_move[1])

        self.skipped_frames += 1
        self.template_position = new_position

        left, top = new_position
        width, height = self.current_report.rect.width, self.current_report.rect.height

        return TrackReport(vot.Rectangle(left, top, width, height), self.current_report.confidence, self.current_report.misc)

    def handle_track_result(self, track_result, misc):
        new_position = track_result.position
        confidence = track_result.confidence

        self.movement_speed = np.append(self.movement_speed, track_result.distance)
        self.last_move = np.array(new_position) - np.array(self.template_position)

        # Reset skipped frames count
        self.skipped_frames = 1

        # Update current orientation
        self.rotation += track_result.rotation
        self.scale *= track_result.scale

        # Update template slice and blur it
        left,top = new_position
        self.template_position = new_position
        right = left + track_result.template_size
        bottom = top + track_result.template_size
        self.set_masked_template(cv2.blur(image[int(top):int(bottom), int(left):int(right)], (3, 3), cv2.BORDER_DEFAULT), track_result.mask)

        # Report region based on binary mask bounding box
        coords = cv2.findNonZero(track_result.mask)
        left, top, width, height = cv2.boundingRect(coords)
        left = left + new_position[0]
        top = top + new_position[1]

        return TrackReport(vot.Rectangle(left, top, width, height), confidence, misc)



    def transform(self, image, rotation, sidelength):
        # Resize the image to the given sidelength
        image = cv2.resize(image, (sidelength, sidelength))

        # Rotate the image given the rotation in degrees
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, rotation, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        return result

    def execute_track(self, image, rotation, scale):

        ### Create the working template ###
        # Get current template size based on scale
        current_template_size = int(self.template_size * self.scale * scale)

        # Transform current template, original template and the template mask 
        transformed_template = self.transform(self.template, rotation, current_template_size)
        transformed_original_template = self.transform(self.original_template, rotation + self.rotation, current_template_size)
        binary_template = self.transform(self.template_mask, self.rotation + rotation, current_template_size)

        # Create the working template based on a current and original template, weighted by the tradeoff parameter
        template = (self.current_to_orignal_template_tradeoff * transformed_template + (1 - self.current_to_orignal_template_tradeoff) * transformed_original_template).astype('uint8')

        if debug_mode:
            cv2.imshow('working template', template)
            #cv2.waitKey(0)

            cv2.imshow('bin', binary_template)
            #cv2.waitKey(0)

        ### Search the working template ###
        # Blur image to match template blur
        blur_image = cv2.blur(image, (3, 3), cv2.BORDER_DEFAULT)

        # Create similarity map
        sim_map = cv2.matchTemplate(blur_image, template, cv2.TM_CCOEFF_NORMED, mask=binary_template)
        sim_map = np.nan_to_num(sim_map, nan=0.0)

        if debug_mode:
            cv2.imshow('simmap', sim_map)
            #cv2.waitKey(0)

        ### Get the best result ###
        # Get top n similarity indices
        rough_indices, values = self.get_max_indices(sim_map, self.n_max_sim_values)

        # Group indices and get a representative for each
        indices, values = self.group_indices(rough_indices, values, self.template_size)

        # Get closest to last frame
        closest, distance, confidence = self.get_closest_indices(indices, values, self.template_position)
        
        # Check if travel distance seems reasonable given the previous observations
        is_reasonable_distance = self.acceptable_distance == 0 or distance < self.acceptable_distance

        return TransformTrackResult(closest, current_template_size, confidence, scale, rotation, distance, is_reasonable_distance, binary_template), (binary_template, self.template, indices, rough_indices)
    
    

    
    def group_indices(self, indices, values, radius):
        """
        Group the indices based on position and values.
        Return a representative for each group with the highest value.
        Indices are grouped if their distance is smaller than radius.

        :param indices: indices, describing positions in a 2d space
        :param values: values of indices
        :param radius: distance for indices to be grouped
        :return: indices and values of representatives of each group
        """ 

        # Create a map of connected indices
        map = np.zeros((len(indices), len(indices)))
        for i, x in enumerate(indices):
            for j, y in enumerate(indices):
                dist = np.linalg.norm(np.array(x) - np.array(y))
                if dist < radius:
                    map[i, j] = 1

        # Create transitive groups of connected indices
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

        # Within each group, get the best representative
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
        """
        Get the closest indices to a position in a 2d space

        :param indices: indices, describing positions in a 2d space
        :param values: values of indices
        :param pos: position to compare indices to
        :return: indices, distance and value of the closest indices
        """ 

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
    
    def get_max_indices(self, value_map, n):
        """
        Get the top n indices with the highest values in a value map

        :param value_map: value map
        :param n: number of indices to return
        :return: indices, distance and value of the closest indices
        """ 

        map = value_map.flatten()
        max_values = np.partition(map, -n)[-n:]
        
        indices = []
        for val in max_values:
            indices.append(np.argwhere(value_map == val)[0])

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
    region, confidence, (mask, template, _, _), (acceptable_distance, indices, rough_indices) = tracker.track(image)

    
    #Use these lines for testing.
    # Comment them when you evaluate with the vot toolkit
    if debug_mode:
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

