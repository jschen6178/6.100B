# 6.100B Problem Set 2 Fall 2023
# Graph Optimization
# Name: Justin Chen
# Collaborators: N/A

# Problem Set 2
# =============
# Finding shortest paths to drive from home to work on a road network

from graph import DirectedRoad, Node, RoadMap


# PROBLEM 2: Building the Road Network
#
# PROBLEM 2.1: Designing your Graph
#
#   What do the graph's nodes represent in this problem? What
#   do the graph's edges represent? Where are the times
#   represented?
#
# Write your answer below as a comment:
#   
#   Graph nodes represent intersections of roads in which
#   a driver may have the choice to drive onto a different road
#   which is represented by an edge connecting two nodes.

#   The times associated with each road are represented as a weight
#   or a cost for traversing the edge.

# PROBLEM 2.2: Implementing create_graph
def create_graph(map_filename):
    """
    Parses the map file and constructs a road map (graph).

    Travel time and traffic multiplier should be each cast to a float.

    Parameters:
        map_filename : str
            Name of the map file.

    Assumes:
        Each entry in the map file consists of the following format, separated by spaces:
            source_node destination_node travel_time road_type traffic_multiplier

        Note: hill road types specified in the text file represent travel time uphill 
              in the source to destination direction. Downhill travel takes 1/3 as long 
              as uphill travel. 

        e.g.
            N0 N1 10 highway 1
        This entry would become two directed roads; one from 'N0' to 'N1' on a highway with
        a weight of 10.0, and another road from 'N1' to 'N0' on a highway using the same weight.

        e.g.
            N2 N3 9 hill 2
        This entry would become two directed roads; one from 'N2' to 'N3' on a hill road with
        a weight of 9.0, and another road from 'N3' to 'N2' on a hill road with a weight of 3.0.
        Note that the directed roads created should both have type 'hill', not 'uphill'/ 'downhill'!

    Returns:
        RoadMap
            A directed road map representing the given map.
    """
    my_roadmap = RoadMap()
    # open the file and read each line
    f = open(map_filename, 'r')
    text_lines = [line.strip(' ').split(' ') for line in f.readlines()]
    
    for line in text_lines:
        #converting data into inputs for node and road
        source_node = Node(line[0])
        destination_node = Node(line[1])
        travel_time = float(line[2])
        road_type = line[3]
        travel_multiplier = float(line[4])
        #inserting the two nodes
        if source_node not in my_roadmap.get_all_nodes():
            my_roadmap.insert_node(source_node)
        if destination_node not in my_roadmap.get_all_nodes():
            my_roadmap.insert_node(destination_node)
        #creating back and forth roads
        road = DirectedRoad(source_node, destination_node, travel_time, road_type, travel_multiplier)
        if road_type == 'hill':
            reverse_road = DirectedRoad(destination_node, source_node, travel_time/3, road_type, travel_multiplier)
        else:
            reverse_road = DirectedRoad(destination_node, source_node, travel_time, road_type, travel_multiplier)
        #adding roads
        my_roadmap.insert_road(road)
        my_roadmap.insert_road(reverse_road)

    return my_roadmap


# PROBLEM 2.3: Testing create_graph
#
#   Go to the bottom of this file, look for the section under FOR PROBLEM 2.3,
#   and follow the instructions in the handout.


# PROBLEM 3: Finding the Shortest Path using Depth-First Search

# Problem 3.1: Objective function
#
#   What is the objective function for this problem? What are the constraints?
#
# Answer:
# The objective of this function is to find the path of least total time
# between two nodes using a depth fir
#

# PROBLEM 3.2: Implement find_shortest_path
def find_shortest_path(roadmap, start, end, restricted_roads=None, has_traffic=False):
    """
    Finds the shortest path between start and end nodes on the road map,
    without using any restricted roads, following traffic conditions.
    If restricted_roads is None, assume there are no restricted roads.
    Use the depth first search algorithm (DFS). 

    Parameters:
        roadmap: RoadMap
            The graph on which to carry out the search.
        start: Node
            Node at which to start.
        end: Node
            Node at which to end.
        restricted_roads: list of str or None
            Road Types not allowed on path. If None, all are roads allowed
        has_traffic: bool
            Flag to indicate whether to get shortest path during traffic or not.

    Returns:
        A two element tuple of the form (best_path, best_time).
            The first item is a list of Nodes, the shortest path from start to end.
            The second item is a float, the length (time traveled) of the best path.
        If there exists no path that satisfies constraints, then return None.
    """
    #dfs algorithm
    def dfs(roadmap, current_path, current_time, end, restricted_roads, has_traffic):
        #if you've reached destination, return the path and time it took
        if current_path[-1] == end:
            return (current_path, current_time)
        #otherwise, start making a list of possible paths from the place you are currently at
        possible_paths = []
        
        #look at all the roads coming out from where you currently are, besides restricted roads
        for next_road in roadmap.get_reachable_roads_from_node(current_path[-1], restricted_roads):
            # if already been on the road, ignore that option
            if next_road.get_destination_node() in current_path:
                continue
            # otherwise, take that road and see where it leads
            new_path = dfs(roadmap, current_path+[next_road.get_destination_node()], current_time+next_road.get_travel_time(has_traffic), end, restricted_roads, has_traffic)
            # if the new path returns something other than empty set, add that to the possible paths 
            if new_path:
                possible_paths.append(new_path)
        # return minimnum of possible paths from this location
        if possible_paths:
            return min(possible_paths, key = lambda x: x[1])
        
    start_path = [start]
    start_time = 0
    return dfs(roadmap, start_path, start_time, end, restricted_roads, has_traffic)


# PROBLEM 4.1: Implement find_shortest_path_no_traffic
def find_shortest_path_no_traffic(filename, start, end):
    """
    Finds the shortest path from start to end during conditions of no traffic.
    Assume there are no restricted roads.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        A two element tuple of the form (best_path, best_time).
            The first item is a list of Nodes, the shortest path from start to end with no traffic.
            The second item is a float, the length (time traveled) of the best path.
        If there exists no path that satisfies constraints, then return None.
    """
    my_roadmap = create_graph(filename)
    return find_shortest_path(my_roadmap, start, end)


# PROBLEM 4.2: Implement find_shortest_path_restricted
def find_shortest_path_restricted(filename, start, end):
    """
    Finds the shortest path from start to end when local roads and hill roads cannot be used.
    Assume no traffic.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        A two element tuple of the form (best_path, best_time).
            The first item is a list of Nodes, the shortest path from start to end given the aforementioned conditions.
            The second item is a float, the length (time traveled) of the best path.
        If there exists no path that satisfies constraints, then return None.
    """
    restricted_roads = ['local', 'hill']
    my_roadmap = create_graph(filename)
    return find_shortest_path(my_roadmap, start, end, restricted_roads)


# PROBLEM 4.3: Implement find_shortest_path_in_traffic
def find_shortest_path_in_traffic(filename, start, end):
    """
    Finds the shortest path from start to end in traffic,
    i.e. when all roads' travel times are multiplied by their traffic multipliers.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        A two element tuple of the form (best_path, best_time).
            The first item is a list of Nodes, the shortest path from start to end given the aforementioned conditions.
            The second item is a float, the length (time traveled) of the best path.
        If there exists no path that satisfies constraints, then return None.
    """
    my_roadmap = create_graph(filename)
    return find_shortest_path(my_roadmap, start, end, has_traffic=True)


if __name__ == '__main__':

    # UNCOMMENT THE LINES BELOW TO DEBUG OR TO EXECUTE PROBLEM 2.3
    pass

    #small_map = create_graph('./maps/small_map.txt')

    # # ------------------------------------------------------------------------
    # # FOR PROBLEM 2.3
    road_map = create_graph("./maps/test_create_graph.txt")
    print(road_map)
    # # ------------------------------------------------------------------------

    # start = Node('N0')
    # end = Node('N4')
    # restricted_roads = []
    # print(find_shortest_path(small_map, start, end, restricted_roads))