import json
import pickle
import random
from itertools import islice

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from Link import Link
from NetworkComponent import NetworkComponent
from environmental_variables import EPOCH_SIZE, STATE_SIZE, NR_MAX_LINKS


class NetworkEngine:

    def __init__(self):
        self.graph_topology = pickle.load(open('small_network.pickle', 'rb'))  # nx.Graph()
        self.links = {}
        self.hosts = {}
        self.switchs = {}
        self.components = {}

        # self.graph_topology = nx.random_internet_as_graph(50)

        self.paths = {}
        self.communication_sequences = {'H1': ['H17', 'H22', 'H15', 'H12', 'H25', 'H13', 'H4', 'H9', 'H10', 'H15'],
                                        'H10': ['', 'H1', 'H8', 'H18', 'H6', 'H20', 'H6', 'H19', '', 'H15'],
                                        'H11': ['H24', 'H14', 'H20', 'H5', 'H17', '', 'H2', 'H3', 'H14', 'H13'],
                                        'H12': ['H1', 'H14', 'H6', 'H7', 'H22', 'H21', 'H18', 'H21', '', 'H23'],
                                        'H13': ['H22', 'H5', 'H4', 'H11', 'H25', 'H8', '', '', 'H11', 'H9'],
                                        'H14': ['H3', '', 'H3', '', '', 'H7', 'H12', 'H3', 'H10', 'H24'],
                                        'H15': ['H1', 'H22', 'H3', '', 'H7', 'H22', '', '', 'H6', 'H12'],
                                        'H16': ['H9', 'H4', '', 'H25', 'H14', 'H10', 'H18', 'H10', 'H18', 'H8'],
                                        'H17': ['H1', 'H4', 'H1', 'H3', 'H9', '', 'H8', '', 'H16', 'H8'],
                                        'H18': ['H5', 'H10', 'H11', 'H8', 'H13', 'H7', 'H21', 'H5', 'H24', 'H10'],
                                        'H19': ['H3', '', 'H13', 'H16', 'H13', '', 'H7', 'H18', '', 'H1'],
                                        'H2': ['H14', '', 'H23', 'H16', '', '', 'H6', 'H20', 'H6', 'H13'],
                                        'H20': ['H1', 'H8', 'H6', 'H10', 'H2', '', 'H1', 'H2', 'H22', 'H7'],
                                        'H21': ['H1', '', 'H9', 'H4', '', 'H15', 'H20', '', 'H9', 'H20'],
                                        'H22': ['', '', 'H3', 'H10', 'H3', 'H6', 'H3', 'H12', 'H11', ''],
                                        'H23': ['', 'H9', 'H12', '', 'H15', 'H25', 'H9', 'H17', 'H12', 'H4'],
                                        'H24': ['', 'H6', 'H11', 'H2', 'H5', 'H7', 'H4', 'H12', 'H3', 'H9'],
                                        'H25': ['', '', 'H6', 'H24', 'H13', 'H19', 'H4', '', 'H21', 'H23'],
                                        'H3': ['H21', 'H4', '', 'H24', 'H10', 'H1', 'H20', 'H13', 'H5', 'H1'],
                                        'H4': ['H1', 'H6', 'H22', 'H8', 'H8', 'H18', 'H18', '', 'H22', 'H13'],
                                        'H5': ['H3', 'H9', 'H1', 'H10', '', 'H2', 'H18', 'H15', 'H1', 'H17'],
                                        'H6': ['H1', '', 'H21', 'H9', 'H10', 'H25', '', 'H2', 'H19', 'H18'],
                                        'H7': ['', 'H1', 'H22', '', 'H9', 'H8', 'H9', '', '', 'H10'],
                                        'H8': ['H1', 'H2', 'H6', 'H12', 'H25', 'H9', 'H19', 'H16', 'H18', 'H20'],
                                        'H9': ['H14', '', 'H12', 'H2', 'H17', 'H15', 'H15', 'H20', 'H5', 'H3']}

        # {'H1': ['H33', 'H49', 'H29', 'H2', 'H48', 'H3', 'H4', 'H16', 'H11', 'H12', 'H42','H20', 'H4', 'H48', 'H8', 'H47', 'H37', 'H21', 'H31', 'H30', 'H44', '', 'H50', 'H16', 'H3', 'H12', 'H13', 'H29', 'H28', ''], 'H2': ['H4', 'H4', 'H37', '', 'H44', 'H18', 'H33', 'H50', 'H38', 'H30', 'H6', 'H46', 'H32', '', 'H36', 'H6', 'H7', 'H33', 'H23', 'H3', 'H42', 'H10', 'H25', 'H28', 'H6', 'H16', 'H11', 'H29', 'H31', 'H29'], 'H3': ['H2', 'H16', 'H47', '', 'H11', '', 'H47', 'H50', 'H26', 'H2', 'H21', 'H28', 'H8', 'H25', '', 'H26', 'H36', 'H48', 'H4', 'H11', '', 'H17', 'H30', 'H5', 'H38', 'H6', 'H38', 'H2', 'H1', 'H36'], 'H4': ['H41', 'H31', 'H13', '', 'H18', '', 'H41', 'H5', 'H35', 'H33', 'H37', 'H36', 'H31', 'H42', 'H34', 'H33', 'H16', 'H23', 'H34', 'H45', 'H24', 'H47', 'H17', 'H47', 'H11', '', 'H40', 'H21', 'H19', 'H25'], 'H5': ['H40', 'H21', 'H7', 'H48', 'H10', '', 'H8', 'H40', 'H13', 'H26', 'H10', 'H4', 'H2', '', 'H41', 'H28', 'H36', '', 'H4', 'H28', '', 'H43', 'H37', 'H37', '', 'H24', 'H43', 'H31', 'H45', 'H9'], 'H6': ['H39', 'H29', '', 'H18', 'H43', 'H3', 'H14', 'H41', 'H19', 'H20', 'H28', '', 'H33', 'H13', '', '', '', 'H4', 'H38', 'H32', 'H8', 'H1', '', 'H39', 'H12', 'H2', 'H48', 'H38', '', 'H50'], 'H7': ['H41', 'H42', 'H1', 'H31', 'H18', 'H35', 'H6', 'H28', 'H25', '', 'H37', 'H19', 'H32', 'H31', 'H12', 'H45', 'H19', 'H12', 'H36', 'H48', 'H29', 'H47', 'H3', 'H32', 'H4', 'H29', 'H46', 'H38', 'H33', 'H20'], 'H8': ['H38', '', 'H45', 'H24', 'H25', 'H22', 'H24', 'H32', 'H36', 'H47', 'H39', 'H30', 'H16', 'H1', 'H40', 'H48', 'H11', 'H48', 'H42', '', 'H23', 'H47', 'H10', 'H18', 'H39', '', 'H43', 'H18', 'H20', 'H30'], 'H9': ['H41', 'H38', 'H30', 'H33', 'H20', 'H32', 'H1', 'H8', 'H20', 'H39', 'H46', '', 'H22', 'H44', 'H42', 'H30', 'H33', 'H16', 'H32', 'H35', 'H25', 'H23', 'H28', 'H17', 'H1', 'H26', 'H11', 'H50', 'H17', 'H32'], 'H10': ['H5', '', 'H29', 'H19', 'H26', 'H32', 'H13', 'H47', 'H45', 'H42', 'H31', 'H46', 'H6', 'H48', '', 'H15', 'H21', 'H19', 'H2', 'H27', 'H11', 'H28', 'H19', 'H28', 'H20', 'H50', 'H14', 'H5', '', 'H24'], 'H11': ['H3', 'H5', 'H23', 'H39', 'H14', 'H13', 'H36', 'H40', 'H50', 'H22', 'H16', 'H47', 'H24', 'H16', 'H4', 'H39', 'H19', 'H30', 'H50', 'H30', 'H40', 'H34', 'H1', '', 'H10', '', 'H49', 'H37', 'H50', 'H24'], 'H12': ['H36', 'H36', 'H23', 'H6', 'H33', 'H5', 'H36', 'H25', 'H15', 'H31', 'H30', 'H37', 'H18', 'H28', 'H5', 'H9', 'H21', 'H45', 'H49', 'H27', '', 'H30', 'H1', 'H13', 'H39', 'H14', '', 'H10', 'H3', 'H30'], 'H13': ['H16', 'H8', 'H31', '', 'H6', 'H40', 'H20', 'H49', 'H22', '', 'H47', 'H35', 'H15', 'H25', '', 'H7', 'H28', 'H38', 'H34', 'H40', 'H25', 'H17', 'H34', '', 'H28', 'H8', 'H9', 'H3', 'H6', 'H34'], 'H14': ['H47', '', 'H32', '', 'H8', 'H44', 'H15', 'H48', 'H3', 'H23', 'H18', 'H44', 'H40', 'H17', 'H30', 'H4', 'H38', 'H48', 'H44', 'H28', '', '', 'H9', 'H24', '', 'H24', '', 'H23', 'H21', 'H43'], 'H15': ['H34', '', 'H30', 'H40', 'H11', 'H16', 'H8', 'H2', 'H12', '', 'H29', 'H18', 'H7', 'H1', 'H10', 'H11', 'H9', 'H36', 'H10', 'H23', 'H39', 'H9', 'H7', 'H10', '', 'H2', 'H12', '', '', 'H21'], 'H16': ['H19', 'H43', 'H5', 'H30', 'H9', 'H33', 'H33', 'H27', 'H17', 'H38', 'H19', 'H22', 'H36', 'H5', '', 'H36', 'H17', 'H19', 'H21', 'H43', 'H49', 'H49', 'H4', 'H30', 'H12', 'H28', 'H25', 'H13', 'H39', ''], 'H17': ['H39', 'H31', 'H44', 'H33', 'H25', 'H4', 'H35', 'H28', 'H35', 'H30', 'H46', 'H25', 'H31', '', 'H8', '', 'H13', 'H44', 'H15', 'H19', 'H1', 'H49', 'H35', 'H36', 'H27', 'H14', 'H36', 'H42', 'H29', 'H40'], 'H18': ['H2', 'H20', 'H48', 'H35', 'H15', 'H16', 'H1', 'H14', 'H9', 'H21', 'H7', 'H15', 'H11', 'H49', 'H9', 'H4', 'H10', 'H42', 'H40', 'H48', 'H45', 'H39', 'H32', 'H7', 'H47', 'H3', 'H28', 'H31', 'H35', 'H49'], 'H19': ['H23', 'H11', 'H38', 'H29', 'H21', 'H43', 'H46', 'H43', 'H34', '', '', 'H50', 'H37', 'H7', 'H4', 'H25', '', 'H39', 'H35', 'H35', 'H7', 'H27', 'H6', 'H16', '', 'H4', 'H44', 'H6', '', 'H39'], 'H20': ['H28', 'H19', '', '', 'H35', 'H46', 'H25', 'H32', 'H40', 'H10', 'H44', 'H35', 'H28', 'H33', 'H29', 'H14', 'H22', 'H5', 'H13', 'H4', '', 'H26', 'H30', 'H22', 'H41', 'H46', 'H16', 'H24', 'H21', 'H15'], 'H21': ['H30', 'H14', 'H20', 'H14', 'H12', 'H49', 'H11', '', 'H41', 'H3', 'H41', 'H11', 'H8', 'H24', 'H7', 'H12', 'H30', '', 'H50', 'H14', 'H44', 'H13', 'H24', 'H42', 'H6', 'H22', 'H31', '', 'H48', 'H2'], 'H22': ['H30', 'H49', 'H16', 'H46', '', 'H16', '', 'H13', 'H13', 'H43', 'H26', 'H47', 'H46', 'H6', 'H31', 'H40', 'H49', 'H9', 'H43', 'H29', 'H32', 'H28', 'H16', 'H3', 'H34', 'H21', 'H7', 'H33', 'H50', 'H21'], 'H23': ['H6', 'H12', 'H3', 'H9', 'H38', 'H24', 'H48', 'H11', 'H33', 'H15', 'H45', 'H18', 'H3', 'H24', 'H29', 'H1', 'H41', 'H39', 'H45', 'H43', 'H12', 'H3', 'H31', 'H6', 'H14', 'H32', 'H39', 'H44', 'H27', ''], 'H24': ['H31', 'H37', 'H13', '', 'H13', 'H34', 'H12', 'H45', 'H33', '', 'H5', 'H13', 'H47', 'H41', 'H46', 'H30', 'H26', 'H50', 'H23', 'H1', 'H34', 'H37', '', 'H27', 'H22', 'H22', 'H43', 'H16', 'H2', 'H48'], 'H25': ['H10', 'H23', 'H44', '', '', 'H32', 'H47', 'H21', 'H30', 'H42', 'H9', '', 'H24', 'H43', 'H4', 'H38', 'H48', '', 'H28', 'H16', 'H2', 'H12', 'H38', 'H1', 'H24', 'H1', '', '', 'H27', 'H30'], 'H26': ['H16', 'H49', 'H9', '', 'H28', 'H10', 'H13', 'H49', 'H6', 'H15', 'H16', 'H35', 'H8', 'H19', 'H5', 'H29', 'H7', 'H16', '', 'H12', 'H2', 'H20', 'H12', 'H25', 'H6', 'H22', 'H35', 'H12', 'H19', 'H45'], 'H27': ['H38', '', 'H35', 'H25', 'H2', '', 'H47', '', 'H5', '', 'H49', 'H8', 'H26', 'H4', 'H17', 'H25', 'H28', 'H15', 'H8', 'H19', 'H33', 'H32', 'H22', '', 'H3', 'H12', 'H11', '', 'H45', 'H36'], 'H28': ['H40', '', 'H2', 'H10', 'H40', 'H10', 'H37', 'H1', 'H2', 'H2', 'H11', 'H29', 'H1', 'H7', 'H29', 'H3', 'H43', 'H18', 'H38', 'H27', 'H39', '', 'H7', 'H3', 'H39', 'H33', 'H14', 'H14', 'H37', 'H30'], 'H29': ['H25', 'H41', 'H45', 'H32', 'H17', '', 'H38', '', 'H19', 'H30', 'H26', 'H7', '', '', 'H16', '', 'H1', 'H8', 'H10', 'H8', '', '', 'H13', 'H13', 'H2', 'H40', 'H16', 'H46', 'H33', 'H24'], 'H30': ['H17', '', 'H47', 'H23', 'H6', '', 'H16', 'H33', '', 'H9', 'H22', 'H21', 'H19', 'H9', 'H34', 'H25', '', 'H48', '', 'H21', 'H49', 'H7', 'H50', 'H35', 'H34', 'H46', 'H19', 'H47', 'H37', 'H4'], 'H31': ['H43', 'H17', 'H4', 'H49', 'H37', 'H23', 'H44', '', 'H37', 'H18', 'H21', 'H4', 'H16', 'H44', '', 'H15', 'H2', 'H35', 'H29', 'H33', 'H49', 'H43', 'H38', '', 'H19', 'H13', 'H40', 'H46', 'H28', 'H8'], 'H32': ['', 'H26', 'H27', 'H37', 'H15', 'H26', 'H23', 'H38', 'H19', 'H6', 'H35', 'H13', 'H29', 'H5', 'H18', 'H25', 'H30', 'H16', 'H35', 'H33', 'H1', 'H38', '', 'H25', 'H14', 'H26', 'H16', 'H30', 'H34', 'H5'], 'H33': ['H48', 'H22', 'H42', 'H24', 'H12', 'H1', 'H6', 'H25', '', 'H47', 'H41', '', 'H15', '', 'H21', 'H34', 'H15', 'H16', 'H10', 'H18', 'H5', 'H28', 'H49', 'H43', 'H6', 'H20', 'H28', 'H50', 'H50', 'H50'], 'H34': ['H23', 'H47', 'H4', 'H26', 'H15', 'H29', 'H44', 'H50', 'H48', 'H37', 'H10', 'H10', 'H32', 'H16', 'H35', '', '', 'H21', 'H1', 'H24', '', 'H29', 'H22', 'H39', 'H15', 'H46', 'H29', 'H33', 'H39', 'H23'], 'H35': ['H43', 'H32', 'H43', 'H7', 'H46', 'H22', 'H37', 'H49', 'H24', 'H32', 'H49', 'H17', 'H8', 'H49', 'H46', 'H11', 'H31', '', '', 'H12', 'H26', 'H6', 'H21', 'H32', 'H49', 'H36', 'H36', 'H8', 'H10', 'H17'], 'H36': ['', 'H13', 'H13', 'H3', 'H45', 'H5', 'H15', 'H28', 'H6', 'H40', 'H5', 'H10', 'H3', 'H46', 'H10', 'H8', 'H2', 'H2', 'H20', 'H16', 'H2', 'H25', 'H50', 'H45', 'H9', 'H26', 'H32', 'H11', 'H30', 'H20'], 'H37': ['H33', 'H32', 'H3', 'H38', '', 'H32', 'H3', 'H26', 'H26', 'H10', 'H20', 'H15', 'H13', 'H36', 'H45', 'H18', 'H29', '', 'H48', 'H39', 'H44', 'H46', 'H31', 'H33', 'H22', 'H45', '', 'H8', 'H13', 'H19'], 'H38': ['H7', 'H40', 'H36', 'H9', 'H35', 'H23', 'H24', 'H19', 'H29', '', 'H7', 'H40', 'H9', 'H36', '', '', 'H48', '', 'H29', 'H48', 'H49', 'H8', 'H13', 'H50', 'H26', 'H25', 'H6', 'H25', 'H14', 'H44'], 'H39': ['H9', 'H47', '', '', 'H36', 'H40', '', 'H9', 'H21', 'H28', 'H48', 'H37', 'H32', 'H23', 'H1', 'H47', 'H45', 'H19', 'H14', 'H30', 'H47', 'H15', 'H9', 'H20', 'H41', 'H21', 'H11', 'H30', 'H6', 'H45'], 'H40': ['H28', 'H17', 'H37', 'H42', 'H35', 'H35', 'H28', 'H22', 'H25', 'H20', 'H30', 'H49', 'H6', 'H2', 'H8', 'H3', 'H8', '', 'H17', 'H27', 'H17', 'H1', 'H24', 'H11', 'H19', 'H22', 'H33', 'H1', 'H16', 'H46'], 'H41': ['H11', 'H12', 'H28', 'H27', 'H34', 'H31', 'H43', 'H20', 'H10', 'H19', 'H18', 'H50', 'H40', 'H28', 'H11', 'H3', 'H33', 'H22', 'H45', 'H1', 'H18', 'H46', 'H3', 'H18', 'H23', 'H32', 'H17', '', 'H1', 'H7'], 'H42': ['H39', 'H37', 'H36', 'H15', 'H18', 'H10', 'H22', 'H5', '', 'H33', 'H3', 'H39', 'H2', 'H9', '', 'H3', 'H25', 'H15', 'H20', 'H8', 'H27', 'H7', 'H19', 'H32', 'H6', 'H36', 'H22', '', 'H30', 'H40'], 'H43': ['H33', 'H16', 'H18', '', 'H32', 'H46', 'H19', 'H48', 'H19', 'H48', 'H22', 'H30', 'H34', 'H46', 'H29', 'H34', 'H41', 'H23', 'H38', 'H25', 'H8', 'H26', '', 'H36', 'H9', 'H41', 'H49', 'H33', 'H29', 'H25'], 'H44': ['H47', 'H17', '', 'H4', 'H16', 'H15', 'H32', 'H6', 'H5', 'H12', 'H41', '', 'H26', 'H17', 'H31', 'H29', 'H3', 'H32', 'H24', 'H35', 'H4', 'H19', 'H22', 'H36', 'H32', 'H13', 'H30', 'H42', 'H13', 'H7'], 'H45': ['H37', 'H29', '', 'H33', 'H18', 'H37', 'H18', 'H20', 'H15', 'H40', 'H35', 'H4', '', 'H20', '', '', 'H34', 'H44', 'H7', 'H13', 'H17', 'H17', 'H50', 'H41', 'H26', '', 'H9', 'H18', '', 'H40'], 'H46': ['H2', '', 'H2', 'H28', 'H36', 'H45', 'H38', 'H17', 'H43', 'H10', 'H2', 'H20', 'H50', 'H11', 'H33', '', 'H8', 'H21', 'H37', 'H19', 'H15', 'H24', 'H30', 'H34', 'H19', 'H27', 'H4', 'H21', 'H32', 'H28'], 'H47': ['H12', 'H30', 'H20', 'H12', 'H32', 'H36', 'H46', '', '', '', '', 'H36', 'H33', 'H45', 'H19', 'H31', 'H32', 'H24', 'H5', 'H7', 'H24', 'H40', 'H8', 'H43', '', 'H7', 'H38', 'H50', 'H23', 'H1'], 'H48': ['H32', 'H1', 'H28', 'H6', 'H23', 'H50', 'H43', 'H7', 'H42', 'H30', 'H32', 'H2', '', 'H4', 'H27', '', 'H7', 'H19', 'H43', 'H21', 'H10', 'H47', 'H8', 'H28', 'H16', 'H40', 'H8', 'H44', 'H16', 'H15'], 'H49': ['H41', 'H32', 'H10', 'H47', 'H38', 'H9', 'H25', 'H42', 'H23', 'H9', 'H27', 'H26', 'H32', '', 'H44', 'H37', 'H40', 'H34', 'H32', 'H38', 'H19', 'H45', 'H31', 'H46', 'H17', 'H32', 'H22', 'H42', 'H10', ''], 'H50': ['H2', 'H8', 'H37', 'H32', 'H31', 'H39', 'H48', 'H25', 'H5', 'H18', 'H46', '', 'H25', 'H30', 'H38', 'H16', 'H23', '', 'H14', 'H21', 'H9', 'H8', 'H23', 'H31', 'H15', 'H7', '', 'H20', 'H19', 'H25']}

        self.create_components(self.graph_topology)

        self.bws = {'H1': 29, 'H2': 28, 'H3': 22, 'H4': 28, 'H5': 33, 'H6': 40, 'H7': 34, 'H8': 29, 'H9': 42, 'H10': 21,
                    'H11': 24, 'H12': 42, 'H13': 34, 'H14': 31, 'H15': 22, 'H16': 26, 'H17': 48, 'H18': 49, 'H19': 50,
                    'H20': 36, 'H21': 34, 'H22': 36, 'H23': 33, 'H24': 24, 'H25': 46, 'H26': 38, 'H27': 38, 'H28': 45,
                    'H29': 21, 'H30': 24, 'H31': 32, 'H32': 50, 'H33': 31, 'H34': 32, 'H35': 49, 'H36': 31, 'H37': 34,
                    'H38': 47, 'H39': 49, 'H40': 29, 'H41': 26, 'H42': 37, 'H43': 28, 'H44': 34, 'H45': 34, 'H46': 43,
                    'H47': 41, 'H48': 24, 'H49': 30, 'H50': 33}

        self.calculate_paths()
        self.hosts = self.get_all_hosts()
        self.number_of_hosts = len(self.hosts)
        self.statistics = {'package_loss': 0, 'package_sent': 0}
        self.single_con_hosts = [f"H{int(host) + 1}" for host in self.graph_topology if
                                 len(self.graph_topology.edges(host)) == 1]

        self.bws = {host: bw if host not in self.single_con_hosts else bw // 3 for host, bw in self.bws.items()}
        print(self.bws)
        print(self.single_con_hosts)

        self.all_tms = json.load(open("all_tms_test.json", mode="r"))
        self.current_index = 0
        self.current_tm_index = self.current_index % EPOCH_SIZE
        self.communication_sequences = self.all_tms[self.current_tm_index]

        """
        self.simulate_communication("H1", "H10", 2, 20)
        self.simulate_communication("H2", "H7", 0, 20)
        self.get_state("H1", 2)
        """

        print("YES")

    def create_components(self, graph: nx.Graph):
        for node in graph.nodes:
            host = f"H{node + 1}"
            if host not in self.components:
                self.components[host] = NetworkComponent(host, self.communication_sequences.get(host, []))

        for edge in graph.edges:
            dst = f"H{edge[1] + 1}"
            origin = f"H{edge[0] + 1}"
            link = Link(origin, dst, 100)
            self.components[origin].add_link(link)
            self.components[dst].add_link(link)
            self.links[link.get_id()] = link

    def build_graph(self):

        for component in self.components.values():
            self.graph_topology.add_node(component.id)

            for neighbor in component.neighbors:
                self.graph_topology.add_edge(component.id, neighbor)

        pp = nx.draw(self.graph_topology, with_labels=True)
        plt.show()

        """
        with open(TOPOLOGY_FILE_NAME, 'r') as topo:
            for line in topo.readlines():
                nodes = line.split()[:2]
                for node in nodes:
                    if not self.graph.has_node(node):
                        self.graph.add_node(node)
                self.graph.add_edge(nodes[0], nodes[1])
        """

    def reset(self, new_tm=False):
        # self.graph_topology = nx.Graph()

        # if True:
        #  self.communication_sequences = generate_traffic_sequence(self)

        self.links = {}
        self.hosts = {}
        self.switchs = {}
        self.components = {}

        if new_tm:
            self.current_tm_index += 1
            self.communication_sequences = self.all_tms[self.current_tm_index % EPOCH_SIZE]

        # print("new ", self.communication_sequences)
        self.create_components(self.graph_topology)
        # self.read_topology("topology_arpanet.txt")
        # self.build_graph()
        # self.calculate_paths()
        self.number_of_hosts = len(self.get_all_hosts())
        self.statistics = {'package_loss': 0, 'package_sent': 0}

        # if new_tm:
        #  self.communication_sequences = generate_traffic_sequence(self)

    def k_shortest_paths(self, graph, source, target, k):
        try:
            calc = list(islice(nx.shortest_simple_paths(graph, source, target), k))
        except nx.NetworkXNoPath:
            calc = []

        final_paths = []

        for p in calc:
            path = []
            for dst in p:
                path.append(f"H{dst + 1}")
            final_paths.append(path)
        return final_paths

    def is_direct_neighbour(self, origin, destination):
        paths = self.get_paths(origin, destination)

        for path in paths:
            if len(path) == 2:
                return True
        return False

    def calculate_paths(self):

        all_hosts = [component for component in self.components if "H" in component]

        for src in all_hosts:

            graph_src = int(src[1:]) - 1
            all_dsts = [h for h in all_hosts if h != src]
            for dst in all_dsts:
                graph_dst = int(dst[1:]) - 1
                self.paths[(src, dst)] = self.k_shortest_paths(self.graph_topology, graph_src, graph_dst, 5)
                self.components[src].set_active_path(dst, 0)
        print(self.paths)
        """   
        for src_host_id in range(1, NUMBER_OF_HOSTS + 1):
            src = "H{}".format(src_host_id)
            for dst_host_id in range(1, NUMBER_OF_HOSTS + 1):
                dst = "H{}".format(dst_host_id)
                if self.graph.has_node(src) and self.graph.has_node(dst):
                    self.paths[(src, dst)] = self.k_shortest_paths(self.graph, src, dst, NUMBER_OF_PATHS)
                    for path in self.paths[(src, dst)]:
                        if len(path) != 0:
                            for i in range(0, len(path)):
                                if "S" in path[i]:
                                    path[i] = path[i].replace("S", "")
                                    path[i] = int(path[i])
        """

    def get_random_dst(self, origin, all_dsts):

        while True:
            dst = random.choice(all_dsts + ['', '', '', '', ''])
            if dst != origin:
                return dst

    def simmulate_turn(self):
        hosts = self.get_all_hosts()

        for host in hosts:
            h = self.components[host]

            if not h.is_busy():
                dst = h.get_dst()

                if dst is None or len(dst) < 2:
                    continue

                h.active_dst = dst
                path_id = h.get_active_path(dst)
                self.simulate_communication(host, dst, path_id, self.bws[host], 2)
                # print(f"SENDING FROM {host} to {dst}")
            else:
                h.update_communication()
                # means the communication is finished
                if not h.is_busy():
                    src = h.id
                    dst = h.current_dst
                    path_chosen = h.get_active_path(dst)

                    if path_chosen >= len(self.paths[(src, dst)]):
                        path_chosen = 0

                    path = self.paths[(src, dst)][path_chosen]
                    self.update_bw_path(path, -h.active_communication_bw)

    def get_nexts_dsts(self):
        return {host: self.components[host].get_next_dst() for host in self.get_all_hosts() if
                not self.components[host].is_busy()}

    def get_busy_hosts(self):
        return [host for host in self.get_all_hosts() if self.components[host].is_busy()]

    def simulate_communication(self, src, dst, path_chosen, bw, nr_turns):
        if path_chosen >= len(self.paths[(src, dst)]):
            path_chosen = 0

        path = self.paths[(src, dst)][path_chosen]
        self.update_bw_path(path, bw)
        self.components[src].set_communication(nr_turns, bw, dst)

    def update_bw_path(self, path, bw):

        origin = path[0]
        destiny = path[-1]
        initial_bw = bw
        update_bw = bw < 0

        for index in range(len(path) - 1):
            src: NetworkComponent
            dst: NetworkComponent
            src = self.components.get(path[index])
            dst = self.components.get(path[index + 1])

            if src.id != origin:
                if bw > 0:
                    src.add_active_communication(origin, destiny)
                else:
                    src.remove_active_communication(origin, destiny)
            else:
                if bw > 0:
                    src.active_dst = destiny
                else:
                    src.active_dst = -1

            link = src.get_link(dst.id)

            if update_bw:
                bw = -1 * link.get_active_communication(origin, destiny)

            link.update_bw(bw)

            if not update_bw:
                link.add_active_communication(origin, destiny, bw)

            if link.bw_available < 0 and not update_bw:
                self.statistics["package_loss"] += -1 * link.bw_available
                bw += link.bw_available
                bw = max(1, bw)

        if not update_bw:
            self.statistics["package_sent"] += bw
            c = self.components[origin]
            c.bw_pct = (bw / initial_bw)
        else:
            c = self.components[origin]
            c.bw_pct = 0

    def read_topology(self, file):
        with open(file, 'r') as topology:
            for row in topology.readlines():
                src_id, dst_id, bw = row.split()
                if src_id not in self.components:
                    self.components[src_id] = NetworkComponent(src_id, self.communication_sequences.get(src_id, []))

                if dst_id not in self.components:
                    self.components[dst_id] = NetworkComponent(dst_id, self.communication_sequences.get(dst_id, []))
                link = Link(src_id, dst_id, bw)

                self.components[src_id].add_link(link)
                self.components[dst_id].add_link(link)
                self.links[link.get_id()] = link
                """
                if 'H' in src_id:
                    host = NetworkComponent(src_id)4
                    self.hosts[src_id] = host #dst_id.replace("S", "")
                    link = Link(src_id, dst_id, bw)
                    host.add_link(link)

                elif 'S' in src_id:
                    src_id = src_id.replace("S", "")
                    dst_id = dst_id.replace("S", "")
                    self.switchs[]
                    self.links[(src_id, dst_id)] = int(bw)
                    self.links[(dst_id, src_id)] = int(bw)
                """

    def get_all_hosts(self):
        return [c for c in self.components if "H" in c]

    def get_link(self, src, dst):
        if (src, dst) in self.links:
            return self.links.get((src, dst))
        elif (dst, src) in self.links:
            return self.links.get((dst, src))
        return None

    def get_paths(self, src, dst):
        key = (src, dst)
        return self.paths.get(key, [])

    def get_min_bw(self, path, n):
        min_bw = 200000000
        for index, component in enumerate(path[:n]):
            src = path[index]
            dst = path[index + 1]
            link = self.get_link(src, dst)
            available = link.get_bw_available_percentage()
            if available < min_bw:
                min_bw = available
        return min_bw

    def get_state(self, host, n=1):

        hostC = self.components.get(host)

        hostC: NetworkComponent
        links = []
        for neighbor in hostC.neighbors:
            links.append(self.get_link(host, neighbor))

        state = np.empty((STATE_SIZE), dtype=object)
        state = np.full((STATE_SIZE), -1)
        link: Link
        for index, link in enumerate(links):
            state[index] = link.get_bw_available_percentage() / 100

        next_dest = hostC.get_next_dst()
        if not next_dest:
            next_dest = -1
        else:
            next_dest = int(next_dest[1:]) / 10

        state[NR_MAX_LINKS] = next_dest

        active_communication = np.array(hostC.get_active_communications()).flatten()
        for index, active in enumerate(active_communication):
            state[index + NR_MAX_LINKS + 1] = active / 10

        state[-1] = self.bws.get(host, 0) / 100

        return state

    def set_active_path(self, host, dsts):
        h = self.components.get(host, None)

        if h is not None:
            for dst, path in dsts.items():
                h.set_active_path(dst, path)
        else:
            print(f"Host {host} not found.")

    def get_link_usage(self):
        bws = [link.get_bw_available_percentage() for link in self.links.values()]
        # return numpry array
        return np.asarray(bws)

    def communication_done(self):
        return all([component.is_done() for name, component in self.components.items() if "H" in name])

def generate_traffic_sequence(network=None):
    if not network:
      network = NetworkEngine()

    hosts = network.get_all_hosts()
    bws = {}
    communications = {}

    for host in hosts:
        bws[host] = random.randint(20, 50)
        for i in range(30):
            dst = network.get_random_dst(host, hosts)
            dsts = communications.get(host, [])
            dsts.append(dst)
            communications[host] = dsts
    #print(communications)
    #print(bws)
    return  communications