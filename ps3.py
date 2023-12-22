# -*- coding: utf-8 -*-
# 6.100B Fall 2023
# Problem Set 3: Robot Simulation
# Name: Justin Chen
# Collaborators:

import math
import random
import matplotlib
import pylab
from ps3_visualize import *


# === Provided class Position, do NOT change
class Position(object):
    """
    A Position represents a location in a two-dimensional room, where
    coordinates are given by floats (x, y).
    """
    
    def __init__(self, x, y):
        """
        Initializes a position with coordinates (x, y).
        """
        self.x = x
        self.y = y

    def get_x(self):
        """
        Returns x coordinate.
        """
        return self.x

    def get_y(self):
        """
        Returns y coordinate.
        """
        return self.y

    def get_new_position(self, angle, speed):
        """
        Computes and returns the new Position after a single clock-tick has
        passed, with this object as the current position, and with the
        specified angle and speed.

        Does NOT test whether the returned position fits inside the room.

        angle: float representing angle in degrees, 0 <= angle < 360
        speed: positive float representing speed

        Returns: a Position object representing the new position.
        """
        old_x, old_y = self.get_x(), self.get_y()

        # Compute the change in position
        delta_y = speed * math.cos(math.radians(angle))
        delta_x = speed * math.sin(math.radians(angle))

        # Add that to the existing position
        new_x = old_x + delta_x
        new_y = old_y + delta_y

        return Position(new_x, new_y)

    def __str__(self):
        return "Position: " + str(math.floor(self.get_x())) + ", " + str(math.floor(self.get_y()))

# === Problem 1
class Room(object):
    """
    A Room represents a rectangular region containing clean or dusty
    tiles.

    A room has a width and a height and contains (width * height) tiles. Each tile
    has some fixed amount of dust. The tile is considered clean only when the amount
    of dust on this tile is 0.
    """

    def __init__(self, width, height, dust_amount):
        """
        Initializes a rectangular room with the specified width, height, and
        dust_amount on each tile.

        width: an integer > 0
        height: an integer > 0
        dust_amount: an integer >= 0
        """
        self.width = width
        self.height = height
        self.dust_amount = dust_amount
        #first index will represent width location, second index will represent height location
        self.tiles = [[dust_amount for i in range(height)] for j in range(width)]

    def get_width(self):
        """
        Returns: an integer; the width of the room
        """
        return self.width

    def get_height(self):
        """
        Returns: an integer; the height of the room
        """
        return self.height

    def get_dust_amount(self, w, h):
        """
        Return the amount of dust on the tile (w, h)

        Assumes that (w, h) represents a valid tile inside the room.

        w: an integer
        h: an integer

        Returns: a float
        """
        return self.dust_amount

    def clean_tile_at_position(self, pos, cleaning_volume):
        """
        Mark the tile under the position pos as cleaned by cleaning_volume amount of dust.

        Assumes that pos represents a valid position inside this room.

        pos: a Position object
        cleaning_volume: a float, the amount of dust to be cleaned in a single time-step.
                  Can be negative which would mean adding dust to the tile.

        Note: The amount of dust on each tile should be NON-NEGATIVE.
              If the cleaning_volume exceeds the amount of dust on the tile, mark it as 0.
        """
        w = math.floor(pos.get_x())
        h = math.floor(pos.get_y())
        #checks if the robot will clean more than what is on the tile
        if self.tiles[w][h] <= cleaning_volume:
            self.tiles[w][h] = 0
        else:
            self.tiles[w][h] -= cleaning_volume
        return

    def is_tile_cleaned(self, w, h):
        """
        Return True if the tile (w, h) has been cleaned.

        Assumes that (w, h) represents a valid tile inside the room.

        w: an integer
        h: an integer

        Returns: True if the tile (w, h) is cleaned, False otherwise

        Note: The tile is considered clean only when the amount of dust on this
              tile is 0.
        """
        return (self.tiles[w][h] == 0)

    def get_num_cleaned_tiles(self):
        """
        Returns: an integer; the total number of clean tiles in the room
        """
        return sum([row.count(0) for row in self.tiles])

    def is_position_in_room(self, pos):
        """
        Determines if pos is inside the room.

        pos: a Position object.
        Returns: True if pos is in the room, False otherwise.
        """
        return (pos.get_x() >= 0 and pos.get_x() < self.get_width() and pos.get_y() >= 0 and pos.get_y() < self.get_height())

    def get_num_tiles(self):
        """
        Returns: an integer; the total number of tiles in the room
        """
        return self.get_width() * self.get_height()

    def get_random_position(self):
        """
        Returns: a Position object; a random position inside the room
        """
        return Position(random.random()*(self.get_width()), random.random()*self.get_height())


class Robot(object):
    """
    Represents a robot cleaning a particular room.

    At all times, the robot has a particular position and direction in the room.
    The robot also has a fixed speed and a fixed cleaning_volume.

    Subclasses of Robot should provide movement strategies by implementing
    update_position_and_clean, which simulates a single time-step.
    """

    def __init__(self, room, speed, cleaning_volume):
        """
        Initializes a Robot with the given speed and given cleaning_volume in the
        specified room. The robot initially has a random direction and a random
        position in the room.

        room:  a Room object.
        speed: a positive float.
        cleaning_volume: a positive float; the amount of dust cleaned by the robot
                  in a single time-step.
        """
        self.room = room
        self.speed = speed
        self.cleaning_volume = cleaning_volume
        #get initial random direction
        self.direction = random.random()*360
        #get initial random position for the robot
        self.position = self.room.get_random_position()

    def get_room(self):
        """
        Returns: the Room the robot is currently in.
        """
        return self.room

    def get_speed(self):
        """
        Returns: a float; the speed of this robot
        """
        return self.speed

    def get_position(self):
        """
        Returns: a Position object giving the robot's position in the room.
        """
        return self.position

    def get_direction(self):
        """
        Returns: a float d giving the direction of the robot as an angle in
        degrees, 0.0 <= d < 360.0.
        """
        return self.direction

    def set_position(self, position):
        """
        Set the position of the robot to position.

        position: a Position object.
        """
        self.position = position

    def set_direction(self, direction):
        """
        Set the direction of the robot to direction.

        direction: float representing an angle in degrees clockwise from north
        """
        self.direction = direction

    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        Moves robot to new position and cleans tile according to robot movement
        rules.
        """
        # DO NOT CHANGE -- implement in subclasses
        raise NotImplementedError


# === Problem 2
class BasicRobot(Robot):
    """
    A BasicRobot is a Robot with the standard movement strategy.

    At each time-step, a BasicRobot attempts to move in its current
    direction; when it would hit a wall, it *instead*
    chooses a new direction randomly.
    """

    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        Calculate the next position for the robot.

        If that position is valid, move the robot to that position. Mark the
        tile it is on as having been cleaned by cleaning_volume amount.

        If the new position is invalid, do not move or clean the tile, but
        rotate once to a random new direction.
        """
        #next_position calculated from speed and direction
        next_position = self.get_position().get_new_position(self.get_direction(), self.get_speed())
        #check if the next location is in the room
        if self.room.is_position_in_room(next_position):
            #if inside room, update the location and clean the next tile
            self.set_position(next_position)
            self.room.clean_tile_at_position(self.position, self.cleaning_volume)
        else:
            #if not inside the room, get another direction
            self.set_direction(random.random()*360)
        return

# === Problem 3
class FaultyRobot(Robot):
    """
    A FaultyRobot is a robot that may accidentally drop dust on a tile. A FaultyRobot will drop some dust
    on the tile it's on with probability p.
    The amount of dropped dust should be a random decimal value between 0 and 0.5.
    Regardless of whether the robot drops dust, it moves to a new position to clean that tile.
    If that new position is not valid, the robot just randomly changes direction.
    """

    p = 0.05

    @staticmethod
    def set_dust_probability(prob):
        """
        Sets the probability of the robot accidentally dropping dust on the tile equal to prob.

        prob: a float (0 <= prob <= 1)
        """
        FaultyRobot.p = prob

    def does_drop_dust(self):
        """
        Answers the question: Does the robot accidentally drop dust on the tile
        at this timestep?
        The robot drops dust with probability p.

        returns: True if the robot drops dust on its tile, False otherwise.
        """
        return random.random() < FaultyRobot.p

    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        1. Before moving, the robot checks if it should drop dust using does_drop_dust.
            If the robot drops dust, it adds a random decimal amount of dust
            between 0 (inclusive) and 0.5 (exclusive).
            If the robot does not drop dust, it just goes to step 2.
        2. Calculate the robot's next position.
            If the position is valid, the robot cleans that tile.
            If the position is not valid, the robot randomly picks a new direction.
        """
        if(self.does_drop_dust()):
            self.room.clean_tile_at_position(self.get_position(), -1*random.random()*0.5)
        #next_position calculated from speed and direction
        next_position = self.get_position().get_new_position(self.get_direction(), self.get_speed())
        #check if the next location is in the room
        if self.room.is_position_in_room(next_position):
            #if inside room, update the location and clean the next tile
            self.set_position(next_position)
            self.room.clean_tile_at_position(self.position, self.cleaning_volume)
        else:
            #if not inside the room, get another direction
            self.set_direction(random.random()*360)
        return


# === Problem 4
class SmartRobot(Robot):
    """
    A SmartRobot is a robot that can decide which direction to go based on the amount of dust it
    sees.

    In one timestep, the SmartRobot will look at its surrounding area and find where there is the
    most dust. In the same timestep, it will move towards one of the positions with the most dust.

    It scans the surrounding area by checking each angles between 0 and 360.
    Naturally, many scans will return the same amount of dust, so after the robot completes it's
    scan, it will randomly pick with uniform probability one of the angles that have the most
    amount of dust.
    """

    def sense_dust_at_angle(self, angle):
        """
        args:
            angle (int): Angle in degrees of which direction to look

        returns:
            The amount of dust at the position the robot would end up if it moved in the direction
            of angle. Returns -1 if the position is a wall.

        DO NOT MODIFY
        """
        lookahead_pos = self.get_position().get_new_position(angle, self.get_speed())
        if not self.get_room().is_position_in_room(lookahead_pos):
            return -1

        tile = (int(lookahead_pos.get_x()), int(lookahead_pos.get_y()))
        return self.get_room().get_dust_amount(tile[0], tile[1])

    def scan_surrounding_area(self):
        """
        Looks at every 5th integer angle from [0, 360) and returns a dictionary mapping angle to
        dust amount.

        If the scanner sees a wall in one of the scans, it will give a reading of -1.

        DO NOT MODIFY
        """
        dust_amounts = {}
        for angle in range(0, 360, 5):
            dust_amounts[angle] = self.sense_dust_at_angle(angle)

        return dust_amounts

    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        Within one time step (i.e. one call to update_position_and_clean), the robot should:

        1. Scan the surrounding area (use scan_surrounding_area)
        2. Find the angles with the maximum amount of dust
        3. Pick one of the dirtiest angles at random and move in that direction (random.choice()
           might be useful!)
        4. Clean the tile the robot lands on
        """
        
        #get the dictinoary from scan surrounding
        dust_amounts = self.scan_surrounding_area()
        #find the maximum dust of the surrounding area
        maximum_dust = max(dust_amounts.values())
        #find the angles that have the maximum dust values
        possible_angles = [angle for angle in list(dust_amounts.keys()) if dust_amounts[angle] == maximum_dust]
        #find and set one angle
 
        self.set_direction(random.choice(possible_angles))

        next_position = self.get_position().get_new_position(self.get_direction(), self.get_speed())
        self.set_position(next_position)
        self.room.clean_tile_at_position(self.position, self.cleaning_volume)
        return
        
        
        


# === Problem 5
def run_simulation(num_robots, speed, cleaning_volume, width, height, dust_amount, min_coverage, num_trials,
                  robot_type):
    """
    Runs num_trials trials of the simulation and returns the mean number of
    time-steps needed to clean the fraction min_coverage of the room.

    The simulation is run with num_robots robots of type robot_type, each
    with the input speed and cleaning_volume in a room of dimensions width x height
    with the dust dust_amount on each tile. Each trial is run in its own Room
    with its own robots.

    num_robots: an int (num_robots > 0)
    speed: a float (speed > 0)
    cleaning_volume: a float (cleaning_volume > 0)
    width: an int (width > 0)
    height: an int (height > 0)
    dust_amount: an int
    min_coverage: a float (0 <= min_coverage <= 1.0)
    num_trials: an int (num_trials > 0)
    robot_type: class of robot to be instantiated (e.g. BasicRobot or
                FaultyRobot)
    """
    
    sum_timesteps = 0
    for i in range(num_trials):
        room = Room(width, height, dust_amount)
        sum_timesteps += run_trial(room, num_robots, robot_type, speed, cleaning_volume, min_coverage)
    return sum_timesteps/num_trials

def run_trial(room, num_robots, robot_type, speed, cleaning_volume, min_coverage):
    timesteps = 0
    #initiate multiple robots
    robots = [robot_type(room, speed, cleaning_volume) for i in range(num_robots)]
    while room.get_num_cleaned_tiles()/room.get_num_tiles() < min_coverage:
        #each robot cleans the room once
        run_time_step(robots)
        timesteps += 1
    return timesteps

def run_time_step(robots):
    #makes each robot update and clean
    for robot in robots:
        robot.update_position_and_clean()
    return

# === Problem 6
#
# ANSWER THE FOLLOWING QUESTIONS:
#
# 1)How does the performance of the three robot types compare when cleaning 80%
#       of a 20x20 room? Describe how performance across robot types changes when
#       more robots are added.
#       The smart robot cleans faster to 80% than the other robots. When more robots are added, this is still true.
#
# 2) How does the performance of the three robot types compare when two of each
#       robot cleans 80% of rooms with dimensions
#       10x30, 20x15, 25x12, and 50x6?
#       Unsure, smart robot seems to be faulty in my code.
#


def show_plot_compare_strategies(title, x_label, y_label):
    """
    Produces a plot comparing the three robot strategies in a 20x20 room with 80%
    minimum coverage.
    """
    num_robot_range = range(1, 6)
    times1 = []
    times2 = []
    times3 = []
    for num_robots in num_robot_range:
        print ("Plotting", num_robots, "robots...")
        times1.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.8, 20, BasicRobot))
        times2.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.8, 20, FaultyRobot))
        times3.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.8, 20, SmartRobot))
    pylab.plot(num_robot_range, times1, 'o-')
    pylab.plot(num_robot_range, times2, 'o-')
    pylab.plot(num_robot_range, times3, 'o-')
    pylab.title(title)
    pylab.legend(('BasicRobot', 'FaultyRobot', 'SmartRobot'))
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()


def show_plot_room_shape(title, x_label, y_label):
    """
    Produces a plot showing dependence of cleaning time on room shape.
    """
    aspect_ratios = []
    times1 = []
    times2 = []
    times3 = []
    for width in [10, 20, 25, 50]:
        height = int(300/width)
        print ("Plotting cleaning time for a room of width:", width, "by height:", height)
        aspect_ratios.append(float(width) / height)
        print("Running for BasicRobot...")
        times1.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, BasicRobot))
        print("Running for FaultyRobot...")
        times2.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, FaultyRobot))
        print("Running for SmartRobot...")
        times3.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, SmartRobot))
    pylab.plot(aspect_ratios, times1, 'o-')
    pylab.plot(aspect_ratios, times2, 'o-')
    pylab.plot(aspect_ratios, times3, 'o-')
    pylab.title(title)
    pylab.legend(('BasicRobot', 'FaultyRobot', 'SmartRobot'), fancybox=True, framealpha=0.5)
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()


if __name__ == "__main__":
    # Test code should go HERE so that test_ps3_student.py doesn't have to run it!
    # All code in this block will be run ONLY if you run ps3.py directly.

    # Uncomment this line to see your implementation of BasicRobot in action!
    # test_robot_movement(BasicRobot, Room)

    # Uncomment this line to see your implementation of FaultyRobot in action!
    # test_robot_movement(FaultyRobot, Room)
    # Uncomment this line to see your implementation of SmartRobot in action!
    test_robot_movement(SmartRobot, Room)

    show_plot_compare_strategies('Time to clean 80% of a 20x20 room, for various numbers of robots','Number of robots','Time (steps)')
    show_plot_room_shape('Time to clean 80% of a 300-tile room for various room shapes','Aspect Ratio', 'Time (steps)')

    pass