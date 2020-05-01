import numpy as np

X = [[0.05,0.1]]
Y = [[0.01,0.99]]


def sigmoid(x):
    return 1/(1+(1/np.power(np.e,x)))

def cost_function (h,y,m):
    sum = 0
    for i in range (m):
        sum += 0.5*np.power(y[i]-h[i],2)
    return sum.sum()

def back_prop (W1,W2,total_eror,pred,y,x,layer1,e):
                  # ([[0.40,0.45],
                  #  [0.50,0.55],
                  #  [0.60,0.60]])
    W1_updated = [[None for i in range (len(W1[0]))] for j in range (len(W1))]
    W2_updated = [[None for i in range (len(W2[0]))] for j in range (len(W2))]
    for i in range (len(W2)-1): #subtract 1 because of bias term
        for j in range (len(W2[0])):
            #partial derivitive of total error with respect to output
            de_do = -(y[j]-pred[j])
            #partial derivitive of ouput with respect to net
            do_dn = pred[j]*(1-pred[j])
            #partial derivitive of net with respect to weight
            dn_dw = layer1[i]
            #print (dn_dw)
            #partial derivitive of error with respect to weight
            de_dw = de_do*do_dn*dn_dw
            #update weight
            W2_updated[i][j] = W2[i][j] - e*de_dw
    #add bias weights back in
    for i in range (len(W2[0])):
        W2_updated[len(W2)-1][i] = W2[len(W2)-1][i]

    for i in range (len(W1)-1):
        for j in range (len(W1[0])):
            #partial derivitive of error of output with respect to net of output
            de_dn1 = -(y[0]-pred[0])*pred[0]*(1-pred[0])
            de_dn2 = -(y[1]-pred[1])*pred[1]*(1-pred[1])
            #partial derivitive of net with respect to out
            dn_do1 = W2[i][j] #idk if this is correct
            dn_do2 = W2[i+1][j]
            #partial derivitive of error with respect to output
            de_do1 = de_dn1*dn_do1
            de_do2 = de_dn2*dn_do2
            #partial derivitive of total error with respect to outptut
            det_do = de_do1+de_do2
            #partial derivitive of out with respect to net
            do_dn = layer1[i]*(1-layer1[i])
            #partial derivitive of net with respect to weight
            dn_dw =x[i]
            #partial derivitive of error with respect to weight
            de_dw = det_do*do_dn*dn_dw
            #update weight
            W1_updated[i][j] = W1[i][j] - e*de_dw
            #print (W1_updated[i][j])
    #add bias weights back in
    for i in range (len(W1[0])):
        W1_updated[len(W1)-1][i] = W1[len(W1)-1][i]
    return W1_updated, W2_updated

def forward_prop(input_node, W1,W2):
    #append bias term
    input_node.append(1)
    layer1 = list(sigmoid(np.dot(input_node,W1)))
    #append bias term
    layer1.append(1)
    layer2 = sigmoid(np.dot(layer1,W2))
    #out = [sigmoid(x) for x in np.dot(hidden_layer1, W1)]
    return layer1, layer2

def train (X,Y):
    W1 = np.array([[0.15,0.20],
                   [0.25,0.30],
                   [0.35,0.35]])
    W2 = np.array([[0.40,0.45],
                   [0.50,0.55],
                   [0.60,0.60]])
    for i in range (10000):
        for x,y in list(zip(X,Y)):
            layer1,pred = forward_prop(x.copy(),W1,W2)
            #print (pred)
            cost = cost_function([pred],Y,len(Y))
            print (cost)
            W1, W2 = back_prop (W1,W2,cost,pred,y,x,layer1,0.5)
            # print (W1)
            # print ("-------------------")
            # print (W2)
            # print ("\n")
train (X,Y)

# W1 = np.array([[0.1,0.1,0.1],[0.1,0.4,0.4],[0.8,0.3,0.3]])
# W2 = np.array([0.1,0.1,0.1,0.1])
