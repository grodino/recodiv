# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:31:43 2017

@author: nbody

Modified by Augustin Godinot @ 2020
"""
import csv
import math
from time import perf_counter
from pathlib import Path

import pandas as pd

from triversity.utils import normalise
from triversity.utils import add_default


class NPartiteGraph:
    def __init__(self, nb_sets):
        """ Create an empty n-partite graph

        :param nb_sets: The number n of distinct sets (or "layers") in the
            n-partite graph
        """

        self.nb_sets = int(nb_sets)
        self.graphs= [[{} for i in range(0, self.nb_sets)] for j in range(0, self.nb_sets)]
        self.linkedTrees= [dict() for i in range(0, self.nb_sets)]
    
        # care it need 2 s its a list of dictionnary (node,linked tree)
        # a linked tree is a dictionnary path -> distribution
        # the path is a tuple
        # a distribution is a dictionnary node -> weight (probability)
        self.res = dict()
        self.saved = set()
        # will save if a result have been spread
        # it will be in res if the result have been computed (all means etc)
        # the save is just for global, not individual
        
        # For renaming (transforming the strings hashes of nodes into integers)
        self.ids = [dict() for i in range(self.nb_sets)]
        self.revids = [dict() for i in range(self.nb_sets)]
        self.last_id = [0 for i in range(self.nb_sets)]

    @classmethod
    def from_folder(cls, folder, n_sets, n_entries=None):
        """Import a graph from a list of files representing the links

        Each line must follow the format : set1_id node1_id set2_id node2_id weight(optional)
        The sets ids must be in [1, 2, ..., n_sets]

        :param folder: string representing the folder containing the dataset
            files
        :param n_sets: The number n of distinct sets (or "layers") in the
            n-partite graph
        :param n_entries: list of line read limits associated to each file. No
            limit at all if n_entries = None or no limit on file i if
            n_entries[i] = 0

        :returns: The created NPartiteGraph graph
        """

        t = perf_counter()

        folder_path = Path(folder)
        file_names = [str(file) for file in folder_path.iterdir() if file.is_file()]
        n_entries = [0] * len(file_names) if n_entries == None else n_entries

        if len(n_entries) != len(file_names):
            raise ValueError('You must provide as much line read limits (n_entries) as data files in the folder')

        graph = cls(n_sets)

        for file_name, n_rows in zip(file_names, n_entries):
            print(f'Reading {n_rows if n_rows > 0 else "all"} rows from file {file_name} ...', end=' ', flush=True)

            if n_rows != 0:
                dataset = pd.read_csv(
                    file_name,
                    sep=' ',
                    names=['node1_level', 'node1', 'node2_level', 'node2', 'weight'],
                    nrows=n_rows
                )
            else:
                dataset = pd.read_csv(
                    file_name,
                    sep=' ',
                    names=['node1_level', 'node1', 'node2_level', 'node2', 'weight']
                )

            print('Done')
            print(f'Importing links from {file_name} ...', end=' ', flush=True)

            for set1_id, node1, set2_id, node2, weight in dataset.itertuples(index=False):
                graph.add_link(set1_id - 1, node1, set2_id - 1, node2, weight, index_node=True)

            print('Done')

        print(f'Dastaset import finished in {perf_counter() - t} s')

        return graph

    def _save_nodes_indexes(self, file_path):
        """Save the nodes indexes in a file

        :param file_path: path to the file where to save the indexes
        """

        with open(file_path, 'w',  newline='') as csvfile:
            writer = csv.writer(csvfile)

            for set_id, nodes in enumerate(self.ids):
                for node_hash, node_id in nodes.items():
                    # to be consistent with dataset files, set ids range from 0
                    # to n_sets, thus set_id + 1
                    writer.writerow([set_id + 1, node_hash, node_id])

    def _recall_node_indexes(self, file_path):
        """Recreates the nodes index and inverse index from a previously
        generated index

        :param file_path: path to the file where to recall the indexes
        """

        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)

            for set_num, node_hash, node_id in reader:
                set_id = int(set_num) - 1
                node_id = int(node_id)

                self.ids[set_id][node_hash] = node_id
                self.revids[set_id][node_id] = node_hash

                if node_id > self.last_id[set_id]:
                    self.last_id[set_id] = node_id

            for set_id, _ in enumerate(self.last_id):
                self.last_id[set_id] += 1

    def _save_indexed_links(self, file_path):
        """Saves the links with the ids of the nodes instead of their hash

        :param file_path: path to the file where to save the links with indexed
            nodes
        """

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            for origin_set_id in range(self.nb_sets):
                # Here the assumption is made that the graph is undirected
                # therefore we save only the upper diagonal of the adjacency matrix
                for destination_set_id in range(origin_set_id, self.nb_sets):
                    for origin_node, neighbors in self.graphs[origin_set_id][destination_set_id].items():
                        for destination_node, weigth in neighbors.items():
                            # to be consistent with dataset files, set ids
                            # range from 0 to n_sets, thus set_id + 1
                            writer.writerow([
                                origin_set_id + 1,
                                origin_node,
                                destination_set_id + 1,
                                destination_node,
                                weigth
                            ])

    def _recall_indexed_links(self, file_path):
        """Recreates the links between ids from a previously generated file

        :param file_path: path to the file where to recall the links with
            indexed nodes
        """

        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)

            for origin_set, origin_node, destination_set, destination_node, weight in reader:
                origin_set = int(origin_set) - 1
                origin_node = int(origin_node)
                destination_set = int(destination_set) - 1
                destination_node = int(destination_node)

                self.add_link(
                    origin_set,
                    origin_node,
                    destination_set,
                    destination_node,
                    int(weight),
                    index_node=False
                )

    def persist(self, folder_path):
        """Persists the nodes indexes and indexed links

        :param folder_path: path to the folder where to persist the files
        """

        folder_path = Path()
        nodes_indexes_path = folder_path.joinpath('nodes_indexes.csv')
        links_path = folder_path.joinpath('links_indexed_nodes.csv')

        print('Persisting nodes indexes ...', end=' ', flush=True)
        self._save_nodes_indexes(nodes_indexes_path)
        print('Done')

        print('Persisting indexed links ...', end=' ', flush=True)
        self._save_indexed_links(links_path)
        print('Done')

    @classmethod
    def recall(cls, folder_path, n_sets):
        """Create a graph from files created by NPartiteGraph.persist

        :param folder_path: path to the folder containing the files created by
            NPartiteGraph.persist()
        :param n_sets: The number n of distinct sets (or "layers") in the
            n-partite graph

        TODO: use pickle ?
        """

        graph = cls(n_sets)

        folder_path = Path(folder_path)
        nodes_indexes_path = folder_path.joinpath('nodes_indexes.csv')
        links_path = folder_path.joinpath('links_indexed_nodes.csv')

        print(f'Recalling nodes index from {nodes_indexes_path} ...', end=' ', flush=True)
        graph._recall_node_indexes(nodes_indexes_path)
        print('Done')

        print(f'Recalling indexed links from {links_path} ...', end=' ', flush=True)
        graph._recall_indexed_links(links_path)
        print('Done')

        return graph

    def get_default_node(self, set_id, node_hash):
        """Get or create the integer id of a node of the set set_id with node name
        node_hash then return it.

        :param set_id: The id of the set (or "layer") in which the node is in
            the n-partite graph
        :param node_hash: the "original" node id (can be anything that can index a dict)

        :returns: The node integer id
        """

        if node_hash in self.ids[set_id].keys():
            return self.ids[set_id][node_hash]
        else:
            x = self.last_id[set_id]
            self.ids[set_id][node_hash] = x
            self.revids[set_id][x] = node_hash
            self.last_id[set_id] += 1
            
            return x
        
    def add_link_d(self, origin_set, origin_node, destination_set, destination_node, weight):
        """Add a directed link in the n-partite graph.

        The nodes must be identified by their int id. If a link between two
        nodes already linked is provided, we sum the weight of the two links

        :param origin_set: The id of the set (or "layer") in which the origin_node is in
            the n-partite graph
        :param origin_node: The id of the origin node
        :param destination_set: The id of the set (or "layer") in which the destination_node is in
            the n-partite graph
        :param destination_node: The id of the destination node
        :param weight: the weight of the link between the two nodes
        """
        d = self.graphs[origin_set][destination_set].setdefault(origin_node, dict())
        add_default(d, destination_node, weight)

    def add_link(self, origin_set, origin_node, destination_set, destination_node, weight, index_node=True):
        """Add an undirected link in the n-partite graph.

        If index_node is True, origin_node and destination_node can be anything that can index a
        dict. If rename is False, origin_node and destination_node must be integers.

        :param origin_set: The id of the set (or "layer") in which the origin_node is in
            the n-partite graph
        :param origin_node: The id or hash of the origin node
        :param destination_set: The id of the set (or "layer") in which the destination_node is in
            the n-partite graph
        :param destination_node: The id or hash of the destination node
        :param weight: the weight of the link between the two nodes
        :param index_node: If True, origin_node and destination_node can be anything that can index a
            dict. If False, origin_node and destination_node must be integers.
        """

        if index_node:
            origin_node_id = self.get_default_node(origin_set, origin_node)
            destination_node_id = self.get_default_node(destination_set, destination_node)
        else:
            origin_node_id = int(origin_node)
            destination_node_id = int(destination_node)

        self.add_link_d(origin_set, origin_node_id, destination_set, destination_node_id, weight)
        self.add_link_d(destination_set, destination_node_id, origin_set, origin_node_id, weight)
    
    def new_tree(self, node, set_id):
        """Create a new tree
        
        :param node: the node at the origin of the tree
        :param set_id: the id of the set (or "layer") of the n-partite graph
            which the node belongs to

        :returns: the newly created  empty tree (a dictionnay)
        """

        self.linkedTrees[set_id][node] = dict()
        
        return self.linkedTrees[set_id][node]
         
    def normalise_all(self):
        """Normalise the weights of the links coming out of each node so that
        the sum of the weights is unitary"""    
        
        for gs in self.graphs:
            for g in gs:
                for d in g.values():
                    normalise(d)
         
    def find_last_saved(self, path):
        """Finds the last step already computed in a sets path

        If path = (1, 2, 3, 6) and we already spread and saved the path (1, 2, 3), this
        function will return the index of the last step that was computed : 2

        :param path: the ids of the sets ("layers") to be traversed (in given order) by the spread.

        :returns: The id of the last sted already computed
        """
        i = 3
        n = len(path)

        while ((i <= n) and (path[:i] in self.saved)):
            i += 1 

        i = i-1
        return i

    def _spread_to_neighbors(self, distrib_in, origin_set, destination_set):  
        """Computes the neighbors distribution.

        Given a given node distribution (probability to arrive at each node), it
        computes the distribution of the neighbors (probability to arrive at
        each neighbor of each node in the original distribution).

        :param distrib_in: a dictionnary with nodes as keys and probability as
            value (distrib_in[node] = probability)
        :param origin_set: The id of the set (or "layer") in which the nodes of 
            distrib_in are
        :param destination_set: The id of the set (or "layer") in which the 
            neighbors of the nodes of distrib_in must be fetched

        :returns: the distribution of the neighbors
        """
        distribution = {}

        for node, probability in distrib_in.items():
            for neighbor, weight in self.graphs[origin_set][destination_set][node].items():
                add_default(distribution, neighbor, probability*weight)
        
        return distribution
        
    def _spread_path(self, last_distribution, path, save_spread, tree=None, complete_path=None, last_saved_path_step=None):
        """Computes the transition probabilities for walks constrained by path.

        More precisely, this computes the probability to go from each node in
        the set path[0] to each node in the set path[-1], traversing the sets path[1:-2]

        :param last_distribution: a dictionnary with nodes as keys and probability as
            value (distrib_in[node] = probability). This represents the transition 
            probabilities from a specific node (unknown by _spread_path)
        :param path: the ids of the sets ("layers") to be traversed (in given
            order) by the spread.
        :param save_spread: boolean. If true, save the distribution of each step in a tree
        :param tree: the memory in which to store the tree (must be a dict)
        :param complete_path: the ids of the sets ("layers") to be traversed (in given
            order) by the spread in its whole.
        :param last_saved_path_step: int, the id of the last path for which we saved the spread

        :returns: the distribution : a dict with dict[node] = probability to
            reach this node via the given path
        """

        if save_spread and (tree == None or complete_path == None or last_saved_path_step == None):
            raise ValueError('If you want to save the spread, you have to provide the args tree, path and i_split.')
        
        distribution = last_distribution
        last_path_set = path[0]
        j = last_saved_path_step #just for save

        for set_id in path[1:]:
            j +=1            
            distribution = self._spread_to_neighbors(distribution, last_path_set, set_id)

            if save_spread:
                tree[path[:j]] = distribution

            last_path_set = set_id

        return distribution
        
    def _diversity_measure(self, distribution):
        """Compute the diversity of a node with the given neighbors distribution.

        To be overridden by the child object.

        :param distribution: a dictionnary with nodes as keys and probability as
            value (distrib_in[node] = probability)

        :returns: a float representing the computed diversity
        """

        raise NotImplementedError()
         
    def _div_init(self, n_nodes_origin, n_nodes_destination):
        """Initialises the "memory" (persistante variables) needed to compute the diversity.

        To be overridden by the child object.

        :param n_nodes_origin: number of nodes in the first set ("layer") of the path
        :param n_nodes_destination : number of nodes in the last set ("layer") of the path
        """

        raise NotImplementedError()
    
    def _div_add(self, node, distribution):
        """Computes the diversity on the given distribution and save it in the given memory.

        To be overridden by the child object.

        :param node: the origin node from which we obtained the transition distribution
        :param distribution: a dictionnary with nodes as keys and probability as
            value (distrib_in[neighbor] = probability)
        """
        #this step is the hard to parallelise
        self._memory[0] += 1
        self._memory[1] += self._diversity_measure(distribution)

        for x in distribution.items():
            add_default(self._memory[2], self._memory[0], self._memory[1])
    
    def _div_return(self):
        """Computes the final result and executes the last operations needed to
        be done after the spread.

        To be overridden by the child object.

        :returns: the final diversity result
        """

        raise NotImplementedError()
            
    def spread_and_divs(self, path, save=True):
        """Spread and compute the diversity via the specified functions (
        _div_init, _div_add, _div_return)

        If already spread just compute the diversity. We use find_last_step just
        one time and not for every nodes.

        :param path: the ids of the sets ("layers") to be traversed (in given order) by the spread.
        :param save: save the results of the diversity calculation
        """
        
        if path in self.res:
            print("Already computed")
            return self.res[path]

        self._div_init(self.last_id[path[0]], self.last_id[path[-1]])
        i_split = self.find_last_saved(path)

        if i_split == 2: #there is no trees
            path0 = path[0]
            path1 = path[1]
            new_path = path[2:]

            for node, neighbors in self.graphs[path0][path1].items():
                if save and len(path) > 2:
                    tree = self.new_tree(node, path0)
                else:
                    tree = None
                
                distribution = self._spread_path(neighbors, path[1:], save, tree, path, i_split)
                self._div_add(node, distribution)
        
        else:
            last_path = path[:i_split]

            print("Already spread until:" + str(last_path))
            for node,tree in self.linkedTrees[path[0]].items():
                d = tree[last_path]
                diversities = self._spread_path(d, path[i_split-1:], save, tree, path, i_split)
                self._div_add(node, diversities)
        
        res = self._div_return()
        self.res[path] = res
        
        if save:
            for j in range(2,len(path)+1):
                self.saved.add(path[:j])       
        
        return res

    def id_to_hash(self, ids, set_id):
        """Converts an id or iterable of ids into a hash or list of hashes

        :param ids: node id or iterable of nodes ids belonging to a same set (or
            "layer")
        :param set_id: id of the set to which the nodes belong

        :returns: hashes corresponding to the songs ids
        """

        if isinstance(ids, int):
            return self.revids[set_id][ids]

        else:
            return [self.revids[set_id][node_id] for node_id in ids]


class IndividualHerfindahlDiversities(NPartiteGraph):
    def _diversity_measure(self, distribution):
        """Compute the Herfindahl diversity of a node with the given neighbors distribution.

        :param distribution: a dictionnary with nodes as keys and probability as
            value (distrib_in[node] = probability)

        :returns: a float representing the computed diversity
        """

        s = sum(x**2 for x in distribution.values())
        diversity = 1 / s if s != 0 else 0

        return diversity
         
    def _div_init(self, n_nodes_origin, n_nodes_destination):
        """Initialises the "memory" (persistante variables) needed to compute the diversity.

        We just want to save the distribution of each nodes for the last step of the path

        :param n_nodes_origin: number of nodes in the first set ("layer") of the path
        :param n_nodes_destination : number of nodes in the last set ("layer") of the path
        """

        self._memory = dict()
    
    def _div_add(self, node, distribution):
        """Saves the distribution of the current step in memory, erasing the previous one

        :param node: the origin node from which we obtained the transition distribution
        :param distribution: a dictionnary with nodes as keys and probability as
            value (distrib_in[neighbor] = probability)
        """
        
        self._memory[node] = distribution
    
    def _div_return(self):
        """Compute the diversity index of each node in the origin set

        :returns: dict of floats
        """
        diversities = {}

        for node, distribution in self._memory.items():
            diversities[node] = self._diversity_measure(distribution)
        
        return diversities

    def diversities(self, path, filename=''):
        """Compute the Herfindal diversity of each node in the first set
        traversed by the path.

        :param path: the ids of the sets ("layers") to be traversed (in given
            order) by the spread.
        :param filename: If not empty, the result is stored in a file with the
            specified name in the 'generated' folder.

        :returns: a dict such that result[node_id] = diversity value
        """

        result = self.spread_and_divs(path, save=True)

        if filename != '':
            file_path = self.generated_folder_path.joinpath(filename)

            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(result.items())

        return result
