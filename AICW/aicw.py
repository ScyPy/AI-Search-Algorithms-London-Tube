import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from queue import PriorityQueue


"""""""""
Reference Used for Uniform Cost Search, check him out.
Github - https://github.com/DeepakKarishetti/Uniform-cost-search#


Run one algorithm at once (comment out unused ones)
"""""""""

#Create new graph
G = nx.Graph()

#Set column namnes
names=["StartingStation", "Destination", "TubeLine", "AverageTimeTaken", "MainZone", "SecondaryZone"]
#Read in data
df = pd.read_csv('tubedata.csv', names=names, quotechar='"').drop_duplicates(inplace=False)

df = df.replace(to_replace='"Heathrow Terminal 3"', value='"Heathrow Terminals 1,2,3"', inplace=False)
df = df.replace(to_replace='"Heathrow Terminals 1', value='"Heathrow Terminals 1,2,3"', inplace=False)

#Replace destinations to Heathrow 1,2,3
df.loc[342]['Destination'] = "Heathrow Terminals 1,2,3"
df.loc[341]['Destination'] = "Heathrow Terminals 1,2,3"

#convert to dictionary
df = df.to_dict('records')

#Preprocess column alignment
df[339]['MainZone'] = ' "6"'
df[339]['TubeLine'] =  ' "Piccadilly"'
df[339]['AverageTimeTaken'] = ' 5'
df[339]['SecondaryZone'] = ' "0"'

#Attribute dictionary
attr = {}

#global values updated in for loop
global tube_line
global main_zone
global secondary_zone
global node
for i in range(0, len(df)):
    #Strip quotation marks
    df[i]['Destination'] = df[i]['Destination'].strip('" "')
    df[i]['AverageTimeTaken'] = df[i]['AverageTimeTaken'].strip('"')
    #Add nodes
    G.add_node(df[i]['StartingStation'])
    #Add edges between startingstation and destination, set weight average time
    G.add_edge(df[i]['StartingStation'], df[i]['Destination'], weight=int(df[i]['AverageTimeTaken']))
    #set node attributes from dict
    main_zone = df[i]['MainZone']
    tube_line = df[i]['TubeLine']
    secondary_zone = df[i]['SecondaryZone']
    node = df[i]['StartingStation']
    destnode = df[i]['Destination']
    #Update attributes to attribute dictionary
    attr.update({node: {"TubeLine":tube_line, "MainZone": main_zone, "SecondaryZone": secondary_zone, "AverageTimeTaken": df[i]['AverageTimeTaken']}})
    #If destination station already in attribute ignore, else add.
    if destnode in attr:
        continue
    else:
        attr.update({destnode: {"TubeLine":tube_line, "MainZone": main_zone, "SecondaryZone": secondary_zone, "AverageTimeTaken": df[i]['AverageTimeTaken']}})
    nx.set_node_attributes(G, attr, name=df[i]['StartingStation'])




#Weighted graphs display edge data
def show_weighted_graph(networkx_graph, node_size, font_size, fig_size):
  plt.figure(num=None, figsize=fig_size, dpi=80)
  plt.axis('off')
  nodes_position = nx.spring_layout(networkx_graph)
  edges_weights  = nx.get_edge_attributes(networkx_graph,'weight')
  nx.draw_networkx_nodes(networkx_graph, nodes_position, node_size=node_size,
                         node_color = ["orange"]*networkx_graph.number_of_nodes())
  nx.draw_networkx_edges(networkx_graph, nodes_position,
                         edgelist=list(networkx_graph.edges), width=2)
  nx.draw_networkx_edge_labels(networkx_graph, nodes_position,
                               edge_labels = edges_weights)
  nx.draw_networkx_labels(networkx_graph, nodes_position, font_size=font_size,
                          font_family='sans-serif')
  plt.axis('off')
  plt.show()



#Color nodes for debugging purposes
color = []
for node in nx.nodes(G):
    if node == 'Mile End':
        color.append("red")
    elif node == 'Stepney Green':
        color.append("red")
    elif node == 'Wembley Park':
        color.append("red")
    else:
        color.append("blue")


#Draw different types of graphs
show_weighted_graph(G, 1500, 15, (10,5))
nx.draw(G, node_color=color, with_labels=True)
plt.show()



#Constructs path from root for DFS,BFS
def construct_path_from_root(node, station_info = True):
    #Fetches all node attributes
    path_from_root = [node['label']]
    #Set current node to parent and add to path
    while node['parent']:
        node = node['parent']
        path_from_root = [node['label']] + path_from_root
    #Print relevant information (set to true)
    if station_info:
        for i in path_from_root:
            print("Station: " + i + "," + " Tubeline: " + attr[i]["TubeLine"]+ "," + " Main Zone: " + attr[i]["MainZone"]+ ","  +" Secondary Zone: " + attr[i]["SecondaryZone"])
    return "Path: {path} Cost: {cost}".format(path=path_from_root, cost = nx.path_weight(G,path_from_root, weight='weight'))


##Construct path from root with UCS (already contains path)
def construct_path_from_root_ucs(node, station_info = True):
    #Fetches the path from node
    path_from_root = node[2]
    #Fetch node attributes from attribute dict for all nodes in path
    if station_info:
        for i in path_from_root:
            print("Station: " + i + "," + " Tubeline: " + attr[i]["TubeLine"]+ "," + " Main Zone: " + attr[i]["MainZone"]+ ","  +" Secondary Zone: " + attr[i]["SecondaryZone"])
    return "Path: {path} Cost: {cost}".format(path=path_from_root, cost = nx.path_weight(G,path_from_root, weight='weight'))



###DFS
def depthFirstSearch(graph, root, goal, compute_exploration_cost=True):

    frontier = [{'label': root, 'parent': None}]
    explored = {root}
    number_of_explored_nodes = 1
    while frontier:
        node = frontier.pop()  # pop from the right of the list
        number_of_explored_nodes += 1
        if node['label'] == goal:
            #nodes expanded
            if compute_exploration_cost:
                print('Expanded Nodes = {}'.format(number_of_explored_nodes))
            return node

        neighbours = list(graph.neighbors(node['label']))
        #loop through neighbours
        for child_label in neighbours:

            child = {'label': child_label, 'parent': node}
            if child_label not in explored:
                #append child to frontier
                frontier.append(child)
                #set child to explored
                explored.add(child_label)
    return None


# DFS Solution
dfs_solution = depthFirstSearch(G, 'Euston', 'Victoria')
dfs_solution = depthFirstSearch(G, 'Canada Water', 'Stratford')
dfs_solution = depthFirstSearch(G, 'New Cross Gate', 'Stepney Green')
dfs_solution = depthFirstSearch(G, 'Ealing Broadway', 'South Kensington')
dfs_solution = depthFirstSearch(G, 'Baker Street', 'Wembley Park')
print("DFS: {0}".format(construct_path_from_root(dfs_solution)))


###BFS
def breadthFirstSearch(graph, root, goal, expanded_nodes=True):
    #if at destination
    if root == goal:
        return None

    #current node is 1
    number_of_explored_nodes = 1
    frontier = [{'label': root, 'parent': None}]
    #Explored Dict, set root default
    explored = {root}

    while frontier:
        #pops from the list
        node = frontier.pop()
        #fetch neighbours of node
        neighbours = list(graph.neighbors(node['label']))
        #for every child neighbour
        for child_label in neighbours:
            child = {'label': child_label, 'parent': node}
            #if child is goal return child
            if child_label == goal:
                #prints nodes expanded
                if expanded_nodes:
                    print('Expanded Nodes = {}'.format(number_of_explored_nodes))
                return child

            if child_label not in explored:
                #Add first in
                frontier = [child] + frontier
                #increment count
                number_of_explored_nodes += 1
                #add to explored
                explored.add(child_label)

    return None


#BFS solution
bfs_solution = breadthFirstSearch(G, 'Euston', 'Victoria')
bfs_solution = breadthFirstSearch(G, 'Canada Water', 'Stratford')
bfs_solution = breadthFirstSearch(G, 'New Cross Gate', 'Stepney Green')
bfs_solution = breadthFirstSearch(G, 'Ealing Broadway', 'South Kensington')
bfs_solution = breadthFirstSearch(G, 'Baker Street', 'Wembley Park')
print("BFS: {0}".format(construct_path_from_root(bfs_solution)))





###UCS
def uniformCostSearch(graph, root, goal, expanded_nodes=True):
    if root == goal:
        return None
    #current node explored is 1
    number_of_explored_nodes = 1
    frontier = []
    #index to compare nodes
    node_index = {}
    #Parent tubeline for extended cost function
    parent_tubeline = attr[root]['TubeLine']
    #Extended Cost Score
    extended_cost = 7
    #[root] stores path
    node = (0, root, [root])
    node_index[node[1]] = [node[0], node[2]]
    #heappush created node
    heapq.heappush(frontier, node)
    #explord no duplicates set
    explored = set()
    while frontier:
        if len(frontier) == 0:
            return None

        node = heapq.heappop(frontier)
        #delete node from index
        del node_index[node[1]]
        if node[1] == goal:
            if expanded_nodes:
                print('Expanded Nodes = {}'.format(number_of_explored_nodes))
            return node
        explored.add(node[1])
        neighbours = graph.neighbors(node[1])
        path = node[2]
        #for every child neighbour
        for child_label in neighbours:
            path.append(child_label)
            node_cost = graph.get_edge_data(child_label, node[1], 'weight')
            #Extend cost function
            childTubeLine = attr[child_label]['TubeLine']
            #Extended cost function if not on same tubeline as parent, add cost
            if parent_tubeline != childTubeLine:
                #Adds extended cost function to our weight
                node_cost['weight'] = int(node_cost['weight']) + extended_cost
                childNode = (node[0] + node_cost['weight'], child_label,path)
            else:
                childNode = (node[0] + node_cost['weight'], child_label, path)

            if child_label not in explored and child_label not in node_index:
                heapq.heappush(frontier, childNode)
                number_of_explored_nodes += 1
                node_index[child_label] = [childNode[0], childNode[2]]
            elif child_label in node_index:
                #if neighbour better than current node in index, assign new
                if childNode[0] < node_index[child_label][0]:
                    node_to_remove = (node_index[child_label][0], child_label, node_index[child_label][1])
                    frontier.remove(node_to_remove)
                    heapq.heapify(frontier)
                    del node_index[child_label]

                    heapq.heappush(frontier, childNode)
                    node_index[child_label] = [childNode[0], childNode[2]]
            path = path[:-1]
    return None


##Ucs solution
ucs_solution = uniformCostSearch(G, 'Euston', 'Victoria')
ucs_solution = uniformCostSearch(G, 'Canada Water', 'Stratford')
ucs_solution = uniformCostSearch(G, 'New Cross Gate', 'Stepney Green')
ucs_solution = uniformCostSearch(G, 'Ealing Broadway', 'South Kensington')
ucs_solution = uniformCostSearch(G, 'Baker Street', 'Wembley Park')
print("UCS: {0}".format(construct_path_from_root_ucs(ucs_solution)))

#Heuristic function
def heuristic(node, goal):
    #Zones of the node and goal
    node_main_zone = attr[node]['MainZone']
    goal_main_zone = attr[goal]['MainZone']

    #Set zones with letters to numbers roughly corresponding to location on map
    if node_main_zone.__contains__("a"):
        node_main_zone = ' "7"'
    elif node_main_zone.__contains__("b"):
        node_main_zone = ' "8"'
    elif node_main_zone.__contains__("c"):
        node_main_zone = ' "9"'
    elif node_main_zone.__contains__("d"):
        node_main_zone = ' "10"'

    if goal_main_zone.__contains__("a"):
        goal_main_zone = ' "7"'
    elif goal_main_zone.__contains__("b"):
        goal_main_zone = ' "8"'
    elif goal_main_zone.__contains__("c"):
        goal_main_zone = ' "9"'
    elif goal_main_zone.__contains__("d"):
        goal_main_zone = ' "10"'
    node_main_zone = node_main_zone.strip(' ""')
    goal_main_zone = goal_main_zone.strip(' ""')

    #Multipliers
    high_multipliers = [7,8,9,10]
    mid_multipliers = [6,5,4]
    low_mulitpliers = [3,2,1]
    ##Get difference between goal zone and node zone && vice versa
    goal_difference = int(goal_main_zone) - int(node_main_zone)
    node_difference = int(node_main_zone) - int(goal_main_zone)

    #Apply multipliers depending on value of goal difference
    if goal_difference > all(high_multipliers):
        score = 9
    elif goal_difference > all(mid_multipliers):
        score = 6
    elif goal_difference > all(low_mulitpliers):
        score = 3
    #Apply multipliers depending on value of node difference
    elif node_difference > all(high_multipliers):
        score = 9
    elif node_difference > all(mid_multipliers):
        score = 6
    elif node_difference > all(low_mulitpliers):
        score = 3
    #If equal cases
    else:
        score = 0
    return score

#A*
def Astar(graph, root, goal):
    #Saved heuristics
    saved_heuristics = {}
    h = heuristic(root, goal)
    saved_heuristics[root] = h
    explored = {}  # This will contain the data of how to get to any node
    explored[root] = (h, [
        root])  # I add the data for the origin node: "Travel cost + heuristic", "Path to get there" and "Admissible Heuristic"

    frontier = PriorityQueue()
    #Add root node, start path and cost
    frontier.put((h, [root], 0))

    #While frontier not empty
    while not frontier.empty():
        # Pop lowest cost element
        _, path, total_cost = frontier.get()
        current_node = path[-1]
        #Fetch all neighbours
        neighbors = graph.neighbors(current_node)
        if current_node == goal:
            return explored[goal], "Nodes Expanded: {0}".format(len(explored))
        #Iterate through neighbours
        for neighbor in neighbors:
            #fetch weight of neighbours
            edge_data = graph.get_edge_data(path[-1], neighbor)
            cost_to_neighbor = edge_data["weight"]

            #check saved heuristics for neighbour else input into heuristic func and save
            if neighbor in saved_heuristics:
                h = saved_heuristics[neighbor]
            else:
                h = heuristic(neighbor, goal)
                saved_heuristics[neighbor] = h

            new_cost = total_cost + cost_to_neighbor
            new_cost_plus_h = new_cost + h
            # If node  never explored, or the cost better than previous cost
            if (neighbor not in explored) or (explored[neighbor][
                                                       0] > new_cost_plus_h):
                next_node = (new_cost_plus_h, path + [neighbor], new_cost)
                #update node with best value
                explored[neighbor] = next_node
                frontier.put(next_node)

    return explored[goal], "Nodes Expanded: ", len(explored)


astarsolution = Astar(G,'Canada Water', 'Stratford')
astarsolution = Astar(G,'New Cross Gate', 'Stepney Green')
astarsolution = Astar(G,'Ealing Broadway', 'South Kensington')
astarsolution = Astar(G,'Baker Street', 'Wembley Park')
print("A*: ",astarsolution)
