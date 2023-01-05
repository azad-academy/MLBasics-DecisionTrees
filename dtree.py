'''
Azad-Academy
Author: J. Rafid Siddiqui
jrs@azaditech.com
https://www.azaditech.com

'''
#==================================================================

from utils import *
import random
import graphviz
from graphviz import Digraph

class Node:

    def __init__(self, constraint = None, text = None , left=None, right=None,  igain=None, label= None):
        
        self.left = left
        self.right = right  
        self.igain = igain  #information gain for the subtree
        self.constraint = constraint   #A string holding the constraint equation (in code format) applied at a node (e.g., X[:,0]<=1)
        self.text = text   #A string holding the text of constraint equation for printing (e.g., X_1<=1)

        self.label = label # Class label at the leaf Nodes


class DecisionTree:
    

    def __init__(self,max_tree_depth=10,min_sample_size=3,method='single'): 

        self.depth = 0
        self.min_features = 2
        self.min_sample_size = min_sample_size
        self.max_tree_depth = max_tree_depth
        self.root = None
        self.num_features = 2

        self.method=method  #Method for constructing the constraint equations
    
    ''' constructs the tree '''
    def tree(self,XY,depth=0):
        
        Y = XY[:,-1]             
        m, n = np.shape(XY[:,:-1])
        self.num_features = n
        
        if(depth>self.depth):
            self.depth = depth

        if m>=self.min_sample_size and depth<self.max_tree_depth:  #stop if no more samples or desired depth is reached
            
            optimum_constraint = self.find_optimum_constraint(XY, m, n)
             
            if optimum_constraint["igain"]>0:
                # build left subtree
                left_subtree = self.tree(optimum_constraint["XY_left"], depth+1)
                # build right subtree
                right_subtree = self.tree(optimum_constraint["XY_right"], depth+1)
                # Create the parent node with above left & righ subtrees
                return Node(optimum_constraint["constraint"],optimum_constraint["text"] , left_subtree, right_subtree, optimum_constraint["igain"])  

        # Leaf Node Reached
        Y = list(Y)
        Y_hat = max(Y, key=Y.count)  #Finding the majority class
        

        return Node(label=Y_hat,text=str(Y_hat))  #Lead Node only has label

    ''' Finding the optimum constraint which maximizes information gain '''
    def find_optimum_constraint(self,XY,m,n):

        X = XY[:,:-1]
        optimum_constraint = {}
        max_igain = -float("inf")
            
        constraints_eqs,constraints_txt = self.build_constraints(X)
        
        # Try all constraints
        for i in range(0,len(constraints_eqs)):
            # Apply the constraint
            XY_left, XY_right = self.apply_constraint(XY, constraints_eqs[i])
            # Subtrees are not empty
            if len(XY_left)>0 and len(XY_right)>0:
                y, left_y, right_y = XY[:, -1], XY_left[:, -1], XY_right[:, -1]
                # computing the information gain
                igain = self.compute_info_gain(y, left_y, right_y, "gini")
                
                # Save the constraint where info-gain is max
                if igain>max_igain:
                    
                    optimum_constraint["constraint"] = constraints_eqs[i]
                    optimum_constraint["text"] = constraints_txt[i]
                    optimum_constraint["XY_left"] = XY_left
                    optimum_constraint["XY_right"] = XY_right
                    optimum_constraint["igain"] = igain
                    max_igain = igain
            #else:
            #    print("subtrees are empty")
                        
        
        return optimum_constraint

    ''' Builds a set of candidate constraint equations '''
    def build_constraints(self, X):
        
        constraints_eqs = []
        constraints_txt = []

        if self.method == "single": # Use one variable only to build constraints, orthognal lines only (e.g., X1 <= c)

            
            for fidx in range(X.shape[1]):
                cc = X[:, fidx]
                cc = np.unique(cc)
                for c in cc:
                    constraints_eqs.append("X[:,{}]<={}".format(fidx,c))
                    constraints_txt.append("X{}<={:.2f}".format(get_subscript(fidx),c))


        elif self.method =="multi": # Use two variables to build slanted line constraints (e.g., X1 <= X2+C)
            indices = range(0,X.shape[1])
            idx_pairs = permutations(indices,2)
            
            for idx_pair in idx_pairs:
                intercepts = X[:,idx_pair[0]] - X[:,idx_pair[1]]
                for intercept in intercepts:
                    constraints_eqs.append("X[:,{}]<=X[:,{}]+{}".format(idx_pair[0],idx_pair[1],intercept))
                    if(intercept<0):
                        constraints_txt.append("X{}<=X{}{:.2f}".format(get_subscript(idx_pair[0]),get_subscript(idx_pair[1]),intercept))    
                    else:
                        constraints_txt.append("X{}<=X{}+{:.2f}".format(get_subscript(idx_pair[0]),get_subscript(idx_pair[1]),intercept))

                # Adding equations for mirror lines with slope -1
                intercepts = X[:,idx_pair[0]] + X[:,idx_pair[1]]
                for intercept in intercepts:
                    constraints_eqs.append("X[:,{}]<=-X[:,{}]+{}".format(idx_pair[0],idx_pair[1],intercept))
                    if(intercept<0):
                        constraints_txt.append("X{}<=-X{}{:.2f}".format(get_subscript(idx_pair[0]),get_subscript(idx_pair[1]),intercept))    
                    else:
                        constraints_txt.append("X{}<=-X{}+{:.2f}".format(get_subscript(idx_pair[0]),get_subscript(idx_pair[1]),intercept))


            #print("Multi-Variable Constraint")
        elif self.method =="non-linear": # Use a non-linear equation (e.g., X1 <= X2.^2 + C)
            indices = range(0,X.shape[1])
            idx_pairs = permutations(indices,2)
            
            for idx_pair in idx_pairs:
                
                for point in X:
                    constraints_eqs.append("(X[:,{}]-{})<=(X[:,{}]-{})*(X[:,{}]-{})".format(idx_pair[0],point[0],idx_pair[1],point[1],idx_pair[1],point[1]))
                    slhs = "(X{}-{:.2f})"
                    sop = "<="
                    srhs = "(X{}-{:.2f}){}"
                    
                    if(point[0]<0):
                        slhs = "(X{}{:.2f})"
                    if(point[1]<0):
                        srhs = "(X{}{:.2f}){}"
                    s = slhs+sop+srhs
                    constraints_txt.append(s.format(get_subscript(idx_pair[0]),point[0],get_subscript(idx_pair[1]),point[1],get_supscript(2)))
                    

        return constraints_eqs,constraints_txt

    ''' Applies a constraint equation and returns a data split '''
    def apply_constraint(self,X,constraint):

        cond = eval(constraint)
        left = X[cond,:]  # Find those points which satisfy the condition
        right = X[np.invert(cond),:] #The points which lie in the opposite Split

        return left, right

    def compute_info_gain(self, parent_node, left_node, right_node, method="entropy"):
                
        W_l = len(left_node) / len(parent_node)
        W_r = len(right_node) / len(parent_node)
        if method=="entropy":
            igain = entropy(parent_node) - (W_l*entropy(left_node) + W_r*entropy(right_node))
        else:
            igain = gini_index(parent_node) - (W_l*gini_index(left_node) + W_r*gini_index(right_node))
            
        return igain

    def train(self, X, Y):
        
        XY = np.concatenate((X, Y), axis=1)
        self.root = self.tree(XY)
    
    def classify(self, X):
        
        Y_hat = np.array([self.predict(np.array([x]), self.root) for x in X]).T
        return Y_hat
    
    def predict(self, X, tree):
       
        if tree.label!=None: return tree.label

        cond = eval(tree.constraint).squeeze()
        if cond:
            return self.predict(X, tree.left)
        else:
            return self.predict(X, tree.right)

    def print_tree(self,scale=1):
        DrawTree(self.root,scale)

    ''' Constructs a GraphViz DiGraph Object '''
    def add_nodes_edges(self, tree, dot=None):
        # Create Digraph object
        if dot is None:
            dot = Digraph()
            dot.node(name=str(tree), label=str(tree.text))

        # Add nodes
        if tree.left:
            dot.node(name=str(tree.left) ,label=str(tree.left.text))
            dot.edge(str(tree), str(tree.left))
            dot = self.add_nodes_edges(tree.left, dot=dot)
            
        if tree.right:
            dot.node(name=str(tree.right) ,label=str(tree.right.text))
            dot.edge(str(tree), str(tree.right))
            dot = self.add_nodes_edges(tree.right, dot=dot)

        return dot
    
    
    
    def show_tree(self,savefile=None):
        
        dot = self.add_nodes_edges(self.root)

        display(dot)

        if(savefile!=None):
            dot.render(savefile,format='png', view=False)
            
    
        
    ''' Plots the constraints on scatter plot '''
    def plot_constraints(self,X):

        if(X is not None):
            if(X.shape[1]>2 or self.num_features>2):
                print("Only works for 2D data")
                return

        fig,axes = plt.subplots(self.depth+1,int(math.pow(2,self.depth)),figsize=(55,20))
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i,j].axis('off')

        if self.root is None:
            return

        queue = [self.root]
        queue.append(None)

        dqueue = [X]
        
        level = 0
         
        root_node_pos_idx = int(math.pow(2,self.depth)/2)
        idx_queue = [root_node_pos_idx]
        left_branch_idx = root_node_pos_idx
        right_branch_idx = root_node_pos_idx

        while len(queue) > 0:

            cur_node = queue.pop(0)
            
            #keeping track of depth level in a BFS traversing
            if(cur_node is None):
                level = level + 1
                queue.append(None)
                if(queue[0] is None):
                    break
            else:
                
                if(cur_node==self.root):
                    self.plot_node_constraint(cur_node,axes[0][root_node_pos_idx],X)
                    
                node_idx = idx_queue.pop(0)                    
                X = dqueue.pop(0)

                if(cur_node.constraint is not None):
                    l_X,r_X = self.apply_constraint(X,cur_node.constraint)

                if cur_node.left is not None:
                    queue.append(cur_node.left)
                    
                    left_branch_idx = node_idx - self.depth + level + 1
                    ax = axes[level+1][left_branch_idx]
                    if(ax.axison):  # Avoiding overlapping
                        if((left_branch_idx-1)>=0):
                            ax = axes[level+1][left_branch_idx-1]
                        else:
                            ax = axes[level+1][left_branch_idx+1]
                    self.plot_node_constraint(cur_node.left,ax,l_X)
                    dqueue.append(l_X)
                    idx_queue.append(left_branch_idx)
                    
                
                if cur_node.right is not None:
                    queue.append(cur_node.right)
                    right_branch_idx = node_idx + self.depth - level + 1
                    ax = axes[level+1][right_branch_idx]
                    if(ax.axison):  #Avoiding overlaps
                        if((right_branch_idx-1)>=0):
                            ax = axes[level+1][right_branch_idx-1]
                        else:
                            ax = axes[level+1][right_branch_idx+1]
                    self.plot_node_constraint(cur_node.right,ax,r_X)
                    dqueue.append(r_X)
                    idx_queue.append(right_branch_idx)
                
        fig.tight_layout()

    
    def plot_node_constraint(self,node,ax,X):
    
        if(len(X)<self.min_sample_size):
            return

        #Plotting scatterplot
        if(node.label is not None): # A leaf Node
            labels = int(node.label)*np.ones((X.shape[0],1),dtype=int)
            plot_data(X,labels.squeeze(),colors=['k','r'],canvas=ax)
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            return
        else:
            labels = 1*eval(node.constraint)
            plot_data(X,labels,colors=['k','r'],canvas=ax)

        


        c = node.constraint.replace("<=","=")
        c_txt = node.text.replace("<=","=")

        
        tokens = c.split("=")
        c_lhs = tokens[0]
        c_rhs = tokens[1]
        
        x0 = np.zeros((X.shape[0],1))
        
        if(c_lhs.find("X[:,0]")<0): #X1 = X0 + C
            x0 = np.array(eval(c_lhs))[:,np.newaxis]
            x1 = np.array(eval(c_rhs))
            if(self.method=="single"):
                x1 = np.ones((X.shape[0],1))*x1
                x0 = np.linspace(min(X[:,0]),max(X[:,0]),X.shape[0])[:,np.newaxis]
        else: #X0 = X1 + C
            x1 = np.array(eval(c_lhs))[:,np.newaxis]
            x0 = np.array(eval(c_rhs))
            if(self.method=="single"):
                x0 = np.ones((X.shape[0],1))*x0
                x1 = np.linspace(min(X[:,1]),max(X[:,1]),X.shape[0])[:,np.newaxis]
                      

        #plotting the constraint
        ax.plot(x0,x1,color='blue',linewidth=3)
        ax.set_ylabel("$X_1$",loc="top")
        ax.set_xlabel("$X_0$",loc="right")
            
        ax.set_title(c_txt)
        ax.axis("on")
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        