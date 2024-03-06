class CorrelationDataset(torch.utils.data.Dataset):
    def __init__(self, netmats, labels, edges, edge_threshold):
        """
        Initializes the dataset with pairwise correlation matrices, labels, and edges.
    
        Parameters:
        - netmats (array-like): Pairwise correlation matrices.
        - edges (array-like): Regions that make up each pairwise correlation from `netmats`.
        - edge_threshold (float64): Minimum Fisher-transformed pairwise correlation value.
        - labels (array-like): Labels indicating gender (0 for male, 1 for female).
        - split_ratio (tuple): Ratios for splitting the data into train, validation, and test sets. Default is (0.8, 0.1, 0.1).
    
        Splits the data into training, validation, and test sets based on the specified split ratios.
        """          
        self.netmats = netmats
        self.labels = labels
        self.edges = edges
        self.edge_threshold = edge_threshold

    # helpers
    def balanced(data, ix_train):
        """Balances the given dataset's training data"""
        balanced_df = data.copy()
        m = data.loc[data['Gender'] == 'M']
        f = data.loc[data['Gender'] == 'F']
        ix_m = np.arange(0, ix_train, 2) # get even indices
        ix_f = np.arange(1, ix_train, 2) # get odd indices
        balanced_df.loc[ix_m, :] = m.iloc[:len(ix_m)].set_index(ix_m) # assign males to even indices
        balanced_df.loc[ix_f, :] = f.iloc[:len(ix_f)].set_index(ix_f) # repeat for females but at odd indices
        return balanced_df

    def create_labels(data):
        """Creates the labels for the given dataset"""
        labels = torch.tensor(data['Gender'].replace({"M":0, "F":1}).values)
        return labels
    
    def create_loaders(self, data, split_ratio):
        self.train_split = split_ratio[0]
        self.val_split = split_ratio[1]
        self.test_split = split_ratio[2]
        tot = len(self)
        train_samples = int(self.train_split * tot)
        val_samples = int(self.val_split * tot)
        test_samples = int(self.test_split * tot)

        # balance the data
        self.data = balanced(data, train_samples)
        self.labels = create_labels(self.data)

        # splitting the data
        self.train_indices = np.arange(0, train_samples) # training data
        self.val_indices = np.arange(train_samples, train_samples + val_samples) # validation data
        self.test_indices = np.arange(train_samples + val_samples, tot) # test data

    def __len__(self):
        """
        Returns the length of entire dataset (train + validate + test)
        """
        return len(self.netmats)

    def __getitem__(self, idx):
        """
        Retrieves the data and label corresponding to the given index.
    
        Parameters:
        - idx (int): Index of the data sample to retrieve.
    
        Returns:
        - graph_data (torch_geometric.data.Data): Graph data containing the correlation matrix and edge indices.
        - label (int): Label indicating the gender (0 for male, 1 for female) of the corresponding data sample.
        """
        x = torch.tensor(self.netmats[idx]).float() # correlation matrix
        edge_index = edge_gen_treshold(self.edges[idx], self.edge_threshold) #Change treshold here
        graph_data = Data(x=x, edge_index=edge_index) 
        label = self.labels[idx]
        return graph_data, label

    def get_split(self, idx):
        """
        Determines the split of a data sample based on its index.
    
        Parameters:
        - idx (int): Index of the data sample.
    
        Returns:
        - split (str): The split of the data sample ('train', 'val', or 'test').
    
        Raises:
        - ValueError: If the index is not found in any of the splits.
        """
        if idx in self.train_indices:
            return 'train'
        elif idx in self.val_indices:
            return 'val'
        elif idx in self.test_indices:
            return 'test'
        else:
            raise ValueError('Index not in any split')

class GraphLoader():
    def __init__():
        pass

    def fit(self, data, edge_threshold, sig=False):
        """
        Creates the dataset with pairwise correlation matrices, labels, and edges.
    
        Parameters:
        - data (array-like): Pairwise correlation matrices.
        - edge_threshold (float64): Minimum Fisher-transformed pairwise correlation value.
        - sig (bool): Whether the edges used will be a subset of the fully connected graph.

        Returns:
        - features (array-like): The node-level features of each graph.
        - edges (array-like): The edges of each graph.
        - labels (array-like): The label of each graph.

        """
        self.data = data
        self.netmats = self.data.netmat # pair-wise correlation matrices
        self.edges = np.argwhere(self.netmats) # fully-connected edges
        if sig:
            self.edges = self.netmats * np.genfromtxt('significant_edges.csv', delimiter=',') # significant edges
        self.labels = self.data.Gender

        self.dataset = CorrelationDataset(netmats, labels, edges, edge_threshold)
        return self.netmats, self.labels, self.edges


    def train_test_split(self, train):
        self.dataset.create_loaders(self.data, train=train)
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, sampler=SubsetRandomSampler(self.dataset.train_indices), collate_fn=self.collate_fn)
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, sampler=SubsetRandomSampler(self.dataset.val_indices), collate_fn=self.collate_fn)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, sampler=SubsetRandomSampler(self.dataset.test_indices), collate_fn=self.collate_fn)
        self.loader_info()
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_loader_info(self, loader):
        size = len(loader) # get the length of the train loader
        f_count = np.sum([graph['label'][0].float() for graph in iter(loader)]) # get the number of f labels in the train loader
        m_count = size - f_count # get the numbr of male labels in the train loader
        print('========================')
        print(f"number of graphs: {size}")
        print(f"number of females: {f_count}")
        print(f"number of males: {m_count}")
        print()
        return

    def loader_info(self):
        print("TRAIN LOADER")
        get_loader_info(self.train_loader)
        print("VAL LOADER")
        get_loader_info(self.val_loader)
        print("TEST LOADER")
        get_loader_info(self.test_loader)
        return
            
# GNN from Application/GNN
class GCN(nn.Module):
    def __init__(self, hidden_d1, hidden_d2, do_p):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(100, hidden_d1)
        self.bn1 = nn.BatchNorm1d(hidden_d1)
        self.conv2 = GCNConv(hidden_d1, hidden_d2)
        self.classifier = Linear(hidden_d2, 1)

    def forward(self, data):
        data = data['graph'][0]
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.tanh(x)
        x = nn.Dropout(p=do_p)(x)
        x = self.conv2(x, edge_index)
        x = torch.tanh(x)
        x = nn.Dropout(p=do_p)(x)
        x = self.classifier(x).mean(dim=0)
        x = torch.sigmoid(x)
        return x

class GNN:
    def __init__(self, hidden_d1, hidden_d2, do_p):
        self.model = GCN(
            hidden_d1=hidden_d1,
            hidden_d2=hidden_d2,
            do_p=do_p
        )

    def train(self, loader):
        self.model.train()
        total_loss = 0
        num_corr = 0
        num_female_guesses = 0
        for d in loader:
            m = d['graph'][0]
            m = m.to(device)
            label = d['label'][0]
            self.optimizer.zero_grad()
            out = self.model(d).squeeze()
            loss = self.criterion(out.float(), label.float())
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            pred = (out.double() > 0.5).float()
            if pred == label.double():
                num_corr += 1
            if pred == 1:
                num_female_guesses += 1 
        return total_loss, num_corr, num_female_guesses

    def eval(self, loader):
        self.model.eval()
        cor = 0
        tot = 0
        tp, fp, tn, fn = 0, 0, 0, 0
        m = 0
        f = 1
        
        for d in loader:
            with torch.no_grad():
                out = self.model(d)
                pred = (out.double() > 0.5).float()

            # metrics pre-liminary calculations
            y = d['label'][0] # actual labels
            tp += (pred == f) & (y == f) # pred: f, actual: f
            fp += (pred == f) & (y == m) # pred: f, actual: m
            tn += (pred == m) & (y == m) # pred: m, actual: m
            fn += (pred == m) & (y == f) # pred: m, actual: f
            cor += (pred == y).sum()
            tot += pred.shape[0]

            # metrics calculations
            prec = tp / (tp + fp) # precision (tp / tp + fp) 
            rec = tp / (tp + fn) # recall (tp / tp + fn)
            acc = cor / tot # accuracy (tp + tn / tp + tn + fp + fn)  
        return prec, rec, acc 

    def fit(self, train_loader, val_loader, criterion, optimizer, num_epochs):
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.losses = np.zeros(num_epochs) 
        self.precs = [np.zeros(num_epochs), np.zeros(num_epochs)] # one list for training, and another for validation
        self.recalls = [np.zeros(num_epochs), np.zeros(num_epochs)] 
        self.accs = [np.zeros(num_epochs), np.zeros(num_epochs)]
        for epoch in range(num_epochs):
            # model training and validation
            t = 0 
            v = 1
            self.losses[epoch], self.num_corr, self.num_female_guesses = train(self.model, train_loader, self.device, self.criterion, self.optimizer)
            self.precs[t][epoch], self.recalls[t][epoch], self.accs[t][epoch] = eval(train_loader)
            self.precs[v][epoch], self.recalls[v][epoch], self.accs[v][epoch] = eval(val_loader)

            self.avg_loss = self.losses[epoch] / len(train_loader)
            # display model performance metrics
            print(f'Epoch: {epoch + 1}/{num_epochs}, '
                f'Loss: {self.losses[epoch]}, '
                f'Avg Loss: {self.avg_loss:.3f}, '
                f'Train: {100 * self.accs[t][epoch]:.2f}%, '
                f'Validation: {100 * self.accs[v][epoch]:.2f}%, '
                f'Num Correct: {self.num_corr}, '
                f'Female Guesses: {self.num_female_guesses} ')
        return
    
    def classification_report(self, loader):
        t = 0
        v = 1
        
        prec, rec, acc = self.eval(loader)
        print('METRICS')
        print('=============')
        print('precision: {prec}')
        print('recall: {rec}')
        print('accuracy: {acc}')

        plt.figure(figsize=(18, 12))
        
        # Plot precisions
        plt.subplot(3, 1, 1)
        plt.plot(self.precs[t], color='blue', label='Training')
        plt.plot(self.precs[v], color='green', label='Validation')
        plt.title('Precisions Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        
        # Plot recalls
        plt.subplot(3, 1, 2)
        plt.plot(self.recalls[t], color='blue', label='Training')
        plt.plot(self.recalls[v], color='green', label='Validation')
        plt.title('Recalls Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracies
        plt.subplot(3, 1, 3)
        plt.plot(self.accs[t], color='blue', label='Training')
        plt.plot(self.accs[v], color='green', label='Validation')
        plt.title('Accuracies Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()



# example usage
graph = GraphLoader()
netmats, labels, edges = graph.fit(data, edge_threshold=0.1, sig=True) # stores the features and labels
train_loader, val_loader, test_loader = graph.train_test_split(split_ratio=(0.8, 0.1, 0.1))

hidden_dim1 = 50
hidden_dim2 = 75
num_epochs =50
gnn = GNN(
    hidden_dim1=hidden_dim1, 
    hidden_dim2=hidden_dim2, 
    dropout_p=0.6,
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(gnn.model.parameters(), lr=0.005)
gnn.fit(train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

# evaluate model
print(gnn.classification_report(test_loader))
