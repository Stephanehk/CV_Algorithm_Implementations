import numpy as np

def sigmoid(x):
    return 1/(1+(1/np.power(np.e,x)))

def cost_function (h,y,m):
    sum = 0
    for i in range (m):
        sum += 0.5*np.power(y[i]-h[i],2)
    return sum.sum()

def back_prop2 (Ws,W2,total_eror,pred,y,x,layers,e):
    W2_updated = [[None for i in range (len(W2[0]))] for j in range (len(W2))]
    for i in range (len(W2)-1): #subtract 1 because of bias term
        for j in range (len(W2[0])):
            #partial derivitive of total error with respect to output
            de_do = -(y[j]-pred[j])
            #partial derivitive of ouput with respect to net
            do_dn = pred[j]*(1-pred[j])
            #partial derivitive of net with respect to weight
            dn_dw = layers[-1][i]
            #print (dn_dw)
            #partial derivitive of error with respect to weight
            de_dw = de_do*do_dn*dn_dw
            #update weight
            W2_updated[i][j] = W2[i][j] - e*de_dw
    #add bias weights back in
    for i in range (len(W2[0])):
        W2_updated[len(W2)-1][i] = W2[len(W2)-1][i]

    Ws_updated = [None for i in range(len(Ws))]
    for m in range (len(Ws)):
        W_updated = [[None for i in range (len(Ws[m][0]))] for j in range (len(Ws[m]))]
        for i in range (len(Ws[m])-1):
            for j in range (len(Ws[m][0])):
                det_do = 0
                for n in range (len(Y[0])):
                    #partial derivitive of error of output with respect to net of output
                    de_dn = -(y[n]-pred[n])*pred[n]*(1-pred[n])
                    #partial derivitive of net with respect to out
                    dn_do = W2[n+1][j]
                    #partial derivitive of error with respect to output
                    de_do = de_dn*dn_do
                    #partial derivitive of total error with respect to outptut
                    det_do+=de_do

                layer = layers[len(Ws)-m-1]
                #print (layer[len(Ws)-m-1][i])
                #partial derivitive of out with respect to net
                do_dn = layer[i]*(1-layer[i])
                #partial derivitive of net with respect to weight
                dn_dw =x[i]
                #partial derivitive of error with respect to weight
                de_dw = det_do*do_dn*dn_dw
                #update weight
                W_updated[i][j] = Ws[m][i][j] - e*de_dw
                #print (W1_updated[i][j])
        #add bias weights back in
        for i in range (len(Ws[m][0])):
            W_updated[len(Ws[m])-1][i] = Ws[m][len(Ws[m])-1][i]
        Ws_updated[m] = W_updated
        return Ws_updated, W2_updated


def back_prop (Ws,W2,total_eror,pred,y,x,layers,e):
    W2_updated = [[None for i in range (len(W2[0]))] for j in range (len(W2))]
    for i in range (len(W2)-1): #subtract 1 because of bias term
        for j in range (len(W2[0])):
            #partial derivitive of total error with respect to output
            de_do = -(y[j]-pred[j])
            #partial derivitive of ouput with respect to net
            do_dn = pred[j]*(1-pred[j])
            #partial derivitive of net with respect to weight
            dn_dw = layers[-1][i]
            #print (dn_dw)
            #partial derivitive of error with respect to weight
            de_dw = de_do*do_dn*dn_dw
            #update weight
            W2_updated[i][j] = W2[i][j] - e*de_dw
    #add bias weights back in
    for i in range (len(W2[0])):
        W2_updated[len(W2)-1][i] = W2[len(W2)-1][i]

    Ws_updated = [None for i in range (len(Ws))]
    #iterate over every hidden layers weights
    for m in range(len(Ws)):
        #start with the last hidden layer
        W = Ws[len(Ws)-m-1]
        layer = layers[len(Ws)-m]

        W_updated = [[None for i in range (len(W[0]))] for j in range (len(W))]
        for i in range (len(W)-1):
            for j in range (len(W[0])):
                #partial derivitive of total error with respect to outptut
                det_do = 0
                for n in range (len(y)):
                    #partial derivitive of error of output with respect to net of output
                    de_dn = -(y[n]-pred[n])*pred[n]*(1-pred[n])
                    #partial derivitive of net with respect to out
                    dn_do = W2[j][n]
                    #partial derivitive of error with respect to output
                    de_do = de_dn*dn_do
                    det_do+=de_do

                #partial derivitive of out with respect to net
                do_dn = layer[i]*(1-layer[i])
                #partial derivitive of net with respect to weight
                dn_dw = layers[len(Ws)-m-1][i] #gets previous layer output
                #dn_dw =x[i]
                de_dw = det_do*do_dn*dn_dw
                #update weight
                W_updated[i][j] = W[i][j] - e*de_dw
                #print (W1_updated[i][j])
        #add bias weights back in
        for i in range (len(W[0])):
            W_updated[len(W)-1][i] = W[len(W)-1][i]
        Ws_updated[len(Ws_updated)-m-1] = W_updated
    return Ws_updated, W2_updated

def forward_prop(input_node,Ws,W2):
    #append bias term
    input_node.append(1)
    layers = [input_node]
    prev_mattrix = input_node
    for W in Ws:
        layer = list(sigmoid(np.dot(prev_mattrix,W)))
        #append bias term
        layer.append(1)
        layers.append(layer)
        prev_mattrix = layer

    out_layer = sigmoid(np.dot(prev_mattrix,W2))
    #out = [sigmoid(x) for x in np.dot(hidden_layer1, W1)]
    return layers, out_layer

def train (X,Y):
    n_nodes = 2
    n_layers =1
    #initialize weights
    Ws = []
    prev_output = len(X[0])
    for i in range (n_layers):
        W = [[0.10 for i in range (n_nodes)] for j in range (prev_output)]
        #all weights need to match the output of the previous layer (len(input) for first hidden layer, len(first hidden layer) for second hidden layer, ...)
        prev_output = n_nodes
        #add bias term weights
        W.append([0.35 for i in range(n_nodes)])
        Ws.append(W)


    W2 = [[0.10 for i in range (len(Y[0]))] for j in range (n_nodes)]
    W2.append([0.60 for i in range(len(Y[0]))])
    W2 = np.array(W2)

    for i in range (10000):
        total_cost = 0
        for x,y in list(zip(X,Y)):
            layers,pred = forward_prop(x.copy(),Ws,W2)
            cost = cost_function(pred,y,len(y))
            total_cost+= cost
            Ws, W2 = back_prop (Ws,W2,cost,pred,y,x,layers,0.5)
        print (total_cost)

#
# X = [[0.05,0.1], [0.15,0.13]]
# Y = [[0.01,0.99],[0.10,1.3]]

X = [[0,1],[1,0],[0,0],[1,1]]
Y = [[0],[0],[0],[1]]

train (X,Y)

# W1 = np.array([[0.1,0.1,0.1],[0.1,0.4,0.4],[0.8,0.3,0.3]])
# W2 = np.array([0.1,0.1,0.1,0.1])
