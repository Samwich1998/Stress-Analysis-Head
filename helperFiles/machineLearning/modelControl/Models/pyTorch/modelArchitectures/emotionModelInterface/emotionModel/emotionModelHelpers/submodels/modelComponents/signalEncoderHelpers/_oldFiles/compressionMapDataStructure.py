# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
from collections import deque

# Pytorch
import torch


# -------------------------------------------------------------------------- #
# ----------------------------- Node Interface ----------------------------- #

class nodeInterface:

    def initializeTree(self, numSignals):
        activeCompressionNodes = []

        # For each signal
        for signalInd in range(numSignals):
            activeCompressionNodes.append(compressionNode())

        return activeCompressionNodes

    def simulateNextLayer(self, currentNodes, numActiveNodes, numBackwardConnections, numForwardConnections):
        nextNodes = []

        # For each incoming node.
        for nodeInd in range(numActiveNodes):
            node = currentNodes[nodeInd]

            # If we are creating a new parent
            if nodeInd % numBackwardConnections == 0:
                forwardNodes = [compressionNode() for _ in range(numForwardConnections)]
                nextNodes.extend(forwardNodes)

            # Add each new parent.
            for forwardNode in forwardNodes:
                node.setforwardNode(forwardNode)

        # Carry over the remaining frozen nodes.
        nextNodes.extend(currentNodes[numActiveNodes:])

        return nextNodes

    def getLayerStateMap(self, currentNodes, device):
        # Initialize holder for the state map.
        stateMap = torch.zeros((len(currentNodes), 2), device=device)

        for nodeInd in range(len(currentNodes)):
            stateMap[nodeInd, 0] = currentNodes[nodeInd].numForwardNodes
            stateMap[nodeInd, 1] = currentNodes[nodeInd].numBackwardNodes

        return stateMap

    def populateForwardNodes(self, lastLayerNodes):
        # Initialize parameters for depth-first search.
        visitedNodes = set()  # Empty initially, will fill with node IDs or references as we visit
        updatingNodes = deque(lastLayerNodes)  # Use deque for efficient popping from the left

        # While we are still updating.
        while updatingNodes:
            updateNode = updatingNodes.popleft()  # Efficient pop from the left

            # Skip if we've already visited this node
            if updateNode in visitedNodes:
                continue

            # Mark the current node as visited
            visitedNodes.add(updateNode)

            # Update the current node.
            updateNode.setForwardNodes()

            # Add the children to update, if not already visited
            for childNode in updateNode.backwardList:
                if childNode not in visitedNodes:
                    updatingNodes.append(childNode)


# -------------------------------------------------------------------------- #
# ---------------------------- Indivisual Node ----------------------------- #

class compressionNode:

    def __init__(self):
        # General parameters.
        self.backwardList = []
        self.forwardList = []

        # Track the compressions/expansions.
        self.numBackwardNodes = 0
        self.numForwardNodes = 0

    def isLeaf(self):
        return len(self.backwardList) == 0

    def setforwardNode(self, forwardNode):
        # Add the parent node to the tree.
        self.forwardList.append(forwardNode)

        # Add this node as a child of the parent.
        forwardNode.backwardList.append(self)
        forwardNode.numBackwardNodes += self.numBackwardNodes + 1

    def setForwardNodes(self):
        for node in self.forwardList:
            self.numForwardNodes += node.numForwardNodes + 1

    def getNodeState(self):
        return self.numForwardNodes, self.numBackwardNodes

# -------------------------------------------------------------------------- #
