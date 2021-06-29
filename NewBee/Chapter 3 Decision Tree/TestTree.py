import trees
import os

"""
For first time to generate Tree
Return decision tree
"""
def GenerateTree():
    f=open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in f.readlines()]
    lensesLabels=['age','prescript','astigmatic','tearRate']
    lensesTree = trees.createTree(lenses, lensesLabels)
    trees.storeTree(lensesTree, "classifierStorage.txt")
    return lensesTree

"""
For load existing decision tree
Return decision tree
"""
def LoadTree():
    return trees.grabTree("classifierStorage.txt")


# Using GenerateTree() to generate tree or using LoadTree() to import tree
# lensesTree = GenerateTree()
lensesTree = LoadTree()

lensesLabels=['age','prescript','astigmatic','tearRate']
label = trees.classify(lensesTree,lensesLabels,["pre","myope","no","normal"])
print("Matched Label is: ", label)
