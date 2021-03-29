# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:31:43 2017

@author: nbody

Modified by Augustin Godinot @ 2020
"""
import csv
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from recodiv.triversity.utils import normalise
from recodiv.triversity.utils import add_default


class NPartiteGraph:
    def __init__(self, nb_sets):
        """ Create an empty n-partite graph

        :param nb_sets: The number n of distinct sets (or "layers") in the
            n-partite graph
        """

        self.nb_sets = int(nb_sets)
        self.graphs= [[{} for i in range(0, self.nb_sets)] for j in range(0, self.nb_sets)]
    
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
                    dtype={
                        'node1_level': np.int8,
                        'node2_level': np.int8,
                        'node1': np.str,
                        'node2': np.str,
                        'weight': np.int32
                    },
                    nrows=n_rows
                )
            else:
                dataset = pd.read_csv(
                    file_name,
                    sep=' ',
                    names=['node1_level', 'node1', 'node2_level', 'node2', 'weight'],
                    dtype={
                        'node1_level': np.int8,
                        'node2_level': np.int8,
                        'node1': np.str,
                        'node2': np.str,
                        'weight': np.int32
                    },
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

    def persist(self, file_path):
        """Persists the graph instance by pickling it

        :param file_path: the path to the pickle file
        """

        print(f'Persisting {self} into {file_path} ...', end=' ', flush=True)
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print('Done')

    @classmethod
    def recall(cls, file_path) -> 'NPartiteGraph':
        """Recreates a graph from a pickled one. WARNING only unpickle files you
        trust !

        :param file_path: the path to the pickle file

        :returns: the recalled graph
        """

        print(f'Recalling {file_path} ...', end=' ', flush=True)
        with open(file_path, 'rb') as file:
            graph = pickle.load(file)
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

        # Fetch origin node already known, else create it
        d = self.graphs[origin_set][destination_set].setdefault(origin_node, dict())

        # Add the neighbor (or update the weight)
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
    
    def normalise_all(self):
        """Normalise the weights of the links coming out of each node so that
        the sum of the weights is unitary"""    
        
        for gs in self.graphs:
            for g in gs:
                for d in g.values():
                    normalise(d)
    
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

            # If node has no neighbor in the next layer (dead end in the path) 
            # we have two options :
            #   - redristribute its weight to the nodes linked to its parent in
            #     the path
            #   - do nothing and accept the the resulting distribution will not
            #     sum to 1
            # For now, we do nothing.
            if node in self.graphs[origin_set][destination_set].keys():
                for neighbor, weight in self.graphs[origin_set][destination_set][node].items():
                    add_default(distribution, neighbor, probability*weight)
            else:
                # print(f'WARNING node {node} not in links ({origin_set} -> {destination_set})')
                pass
        
        return distribution
        
    def _spread_path(self, last_distribution, path):
        """Computes the transition probabilities for walks constrained by path.

        More precisely, this computes the probability to go from each node in
        the set path[0] to each node in the set path[-1], traversing the sets path[1:-2]

        :param last_distribution: a dictionnary with nodes as keys and probability as
            value (distrib_in[node] = probability). This represents the transition 
            probabilities from a specific node (unknown by _spread_path)
        :param path: the ids of the sets ("layers") to be traversed (in given
            order) by the spread.

        :returns: the distribution : a dict with dict[node] = probability to
            reach this node via the given path
        """

        distribution = last_distribution
        last_set_id = path[0]

        for set_id in path[1:]:
            distribution = self._spread_to_neighbors(distribution, last_set_id, set_id)

            last_set_id = set_id

        return distribution
            
    def spread_and_divs(self, path, progress=True):
        """Spread and compute the diversity via the specified functions (
        _div_init, _div_add, _div_return)

        If already spread just compute the diversity. We use find_last_step just
        one time and not for every nodes.

        :param path: the ids of the sets ("layers") to be traversed (in given order) by the spread.
        :param progress: show the progress via tqdm
        """
        
        if path in self.res:
            print("Already computed")
            return self.res[path]

        self._div_init(self.last_id[path[0]], self.last_id[path[-1]])

        for node, neighbors in tqdm(self.graphs[path[0]][path[1]].items(), disable=not(progress), desc='Computing diversities'):            
            distribution = self._spread_path(neighbors, path[1:])
            self._div_add(node, distribution)
        
        res = self._div_return()
        self.res[path] = res

        return res

    def spread_node(self, node_hash, path):
        """Compute the distributions of reached nodes in the layer path[-1] for
        a single node in the layer path[0] through the given layers layer[1: -1]
        
        :param node_hash: the "name" of the node to compute 
        :param path: the ids of the sets ("layers") to be traversed (in given order) by the spread.
        
        :returns: The reached node distribution
        """
        node_id = self.ids[path[0]][node_hash]
        neighbors = self.graphs[path[0]][path[1]][node_id]

        distribution = self._spread_path(neighbors, path[1:])

        return {self.revids[path[-1]][node_id]: value for node_id, value in distribution.items()}

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

    def n_links(self, from_set, to_set):
        """Returns the number of links from as layer to an other
        
        :param from_set: id of the origin layer
        :param to_set: id of the destination layer
        """

        n = 0

        for node, neigbours in self.graphs[from_set][to_set].items():
            n += len(neigbours)

        return n



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

    def diversities(self, path, progress=True):
        """Compute the Herfindal diversity of each node in the first set
        traversed by the path.

        :param path: the ids of the sets ("layers") to be traversed (in given
            order) by the spread.

        :returns: a dict such that result[node_hash] = diversity value
        """

        result = {}
        result_by_id = self.spread_and_divs(path, progress=progress)

        for node_id, diversity in result_by_id.items():
            result[self.revids[path[0]][node_id]] = diversity

        del result_by_id

        return result
