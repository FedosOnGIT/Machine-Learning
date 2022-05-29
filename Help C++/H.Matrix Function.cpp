#include<bits/stdc++.h>

using namespace std;

struct Matrix {
    Matrix() {
        rows = 0;
        columns = 0;
        values = vector<vector<double>>();
    }

    Matrix(int rows, int columns, double fill) {
        this->rows = rows;
        this->columns = columns;
        values = vector<vector<double>>();
        for (int i = 0; i < rows; i++) {
            values.emplace_back(columns, fill);
        }
    }

    Matrix(int rows, int columns, vector<vector<double>> &values) {
        this->rows = rows;
        this->columns = columns;
        this->values = values;
    }

    vector<double> operator[](int row) {
        assert(row < rows);
        return values[row];
    }

    Matrix operator()(const function<double(double)> &f) {
        vector<vector<double>> applied;
        for (int i = 0; i < rows; i++) {
            vector<double> row;
            for (int j = 0; j < columns; j++) {
                row.push_back(f(values[i][j]));
            }
            applied.push_back(row);
        }
        return {rows, columns, applied};
    }

    int rows;
    int columns;
    vector<vector<double>> values;
};

Matrix operator!(Matrix &transposing) {
    vector<vector<double>> transposed;
    for (int i = 0; i < transposing.columns; i++) {
        vector<double> row;
        for (int j = 0; j < transposing.rows; j++) {
            row.push_back(transposing[j][i]);
        }
        transposed.push_back(row);
    }
    return {transposing.columns, transposing.rows, transposed};
}

Matrix operator+(Matrix &first, Matrix &second) {
    assert(first.rows == second.rows && first.columns == second.columns);
    int rows = first.rows;
    int columns = first.columns;
    vector<vector<double>> values;
    for (int i = 0; i < rows; i++) {
        vector<double> row;
        for (int j = 0; j < columns; j++) {
            row.push_back(first[i][j] + second[i][j]);
        }
        values.push_back(row);
    }
    return {rows, columns, values};
}

Matrix operator-(Matrix &first, Matrix &second) {
    assert(first.rows == second.rows && first.columns == second.columns);
    int rows = first.rows;
    int columns = first.columns;
    vector<vector<double>> values;
    for (int i = 0; i < rows; i++) {
        vector<double> row;
        for (int j = 0; j < columns; j++) {
            row.push_back(first[i][j] - second[i][j]);
        }
        values.push_back(row);
    }
    return {rows, columns, values};
}

Matrix operator*(Matrix &first, Matrix &second) {
    assert(first.columns == second.rows);
    int rows = first.rows;
    int columns = second.columns;
    vector<vector<double>> values;
    for (int i = 0; i < rows; i++) {
        vector<double> row;
        for (int j = 0; j < columns; j++) {
            double value = 0;
            for (int k = 0; k < first.columns; k++) {
                value += first[i][k] * second[k][j];
            }
            row.push_back(value);
        }
        values.push_back(row);
    }
    return {rows, columns, values};
}

Matrix operator&(Matrix &first, Matrix &second) {
    assert(first.rows == second.rows && first.columns == second.columns);
    int rows = first.rows;
    int columns = first.columns;
    vector<vector<double>> values;
    for (int i = 0; i < rows; i++) {
        vector<double> row;
        for (int j = 0; j < columns; j++) {
            row.push_back(first[i][j] * second[i][j]);
        }
        values.push_back(row);
    }
    return {rows, columns, values};
}

ostream &operator<<(ostream &os, Matrix &out) {
    for (int i = 0; i < out.rows; i++) {
        for (int j = 0; j < out.columns; j++) {
            os << fixed << setprecision(9) << out[i][j] << " ";
        }
        os << "\n";
    }
    return os;
}

bool operator==(Matrix &first, Matrix &second) {
    if (first.rows == second.rows && first.columns == second.columns) {
        for (int i = 0; i < first.rows; i++) {
            for (int j = 0; j < first.columns; j++) {
                if (first[i][j] != second[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
    return false;
}

struct Graph;

struct Node {

    explicit Node(vector<int> &parents) {
        this->parents = parents;
        matrix = Matrix();
    }

    virtual void calculate(Graph &graph) = 0;

    virtual Matrix derivative(Graph &graph, int index) = 0;

    void init_gradient() {
        gradient = Matrix(matrix.rows, matrix.columns, 0);
    }

    vector<int> parents;
    Matrix matrix;
    Matrix gradient;
};

struct Graph {
    Graph() {
        nodes = vector<Node *>();
    }

    void add(Node *node) {
        nodes.push_back(node);
    }

    Node &operator[](int index) {
        assert(index < nodes.size());
        return *nodes[index];
    }

    vector<Node *> nodes;
};

struct Var : Node {
    explicit Var(vector<int> &parents) : Node(parents) {
        assert(parents.empty());
    }

    void calculate(Graph &graph) override {}

    Matrix derivative(Graph &graph, int index) override {
        return gradient;
    }
};

struct Tanh : Node {
    explicit Tanh(vector<int> &parents) : Node(parents) {
        assert(parents.size() == 1);
    }

    void calculate(Graph &graph) override {
        matrix = graph[parents[0]].matrix([](double x) {
            return tanh(x);
        });
    }

    Matrix derivative(Graph &graph, int index) override {
        Matrix one = Matrix(matrix.rows, matrix.columns, 1);
        Matrix square = matrix & matrix;
        Matrix first = (one - square);
        return first & gradient;
    }
};

struct ReLu : Node {
    ReLu(vector<int> &parents, double alpha) : Node(parents) {
        assert(parents.size() == 1);
        this->alpha = 1 / alpha;
    }

    void calculate(Graph &graph) override {
        double alpha_parameter = alpha;
        matrix = graph[parents[0]].matrix([alpha_parameter](double x) {
            if (x >= 0) {
                return x;
            } else {
                return x * alpha_parameter;
            }
        });
    }

    Matrix derivative(Graph &graph, int index) override {
        double alpha_parameter = alpha;
        Matrix derivative = graph[parents[index]].matrix([alpha_parameter](double x) {
            if (x >= 0) {
                return (double) 1;
            }
            return alpha_parameter;
        });
        return derivative & gradient;
    }

    double alpha;
};

struct Mul : Node {
    explicit Mul(vector<int> &parents) : Node(parents) {
        assert(parents.size() == 2);
    }

    void calculate(Graph &graph) override {
        Matrix first = graph[parents[0]].matrix;
        Matrix second = graph[parents[1]].matrix;
        matrix = first * second;
    }

    Matrix derivative(Graph &graph, int index) override {
        Matrix transposed = index == 0 ?
                            !graph[parents[1]].matrix :
                            !graph[parents[0]].matrix;
        return index == 0?
               gradient * transposed :
               transposed * gradient;
    }
};

struct Sum : Node {
    explicit Sum(vector<int> &parents) : Node(parents) {
        assert(!parents.empty());
    }

    void calculate(Graph &graph) override {
        int rows = graph[parents[0]].matrix.rows;
        int columns = graph[parents[0]].matrix.columns;
        matrix = Matrix(rows, columns, 0);
        for (int parent: parents) {
            matrix = matrix + graph[parent].matrix;
        }
    }

    Matrix derivative(Graph &graph, int index) override {
        return gradient;
    }
};

struct Had : Node {
    explicit Had(vector<int> &parents) : Node(parents) {
        assert(!parents.empty());
    }

    void calculate(Graph &graph) override {
        int rows = graph[parents[0]].matrix.rows;
        int columns = graph[parents[0]].matrix.columns;
        matrix = Matrix(rows, columns, 1);
        for (int parent: parents) {
            matrix = matrix & graph[parent].matrix;
        }
    }

    Matrix derivative(Graph &graph, int index) override {
        Matrix result = gradient;
        for (int i = 0; i < parents.size(); i++) {
            if (i != index) {
                result = result & graph[parents[i]].matrix;
            }
        }
        return result;
    }
};

void calculate(Graph &graph, int last) {
    for (Node *node: graph.nodes) {
        node->calculate(graph);
    }
    int size = (int) graph.nodes.size();
    for (int i = size - last; i < size; i++) {
        cout << graph[i].matrix;
    }
}

void derivative(Graph &graph, int first) {
    for (int i = (int) graph.nodes.size() - 1; i >= 0; i--) {
        for (int index = 0; index < graph[i].parents.size(); index++) {
            Matrix plus = graph[i].derivative(graph, index);
            graph[graph[i].parents[index]].gradient = graph[graph[i].parents[index]].gradient + plus;
        }
    }
    for (int i = 0; i < first; i++) {
        cout << graph[i].gradient;
    }
}

int main() {
    int n, m, k;
    cin >> n >> m >> k;
    Graph graph;
    vector<pair<int, int> > inputs;
    for (int i = 0; i < n; i++) {
        string operation;
        cin >> operation;
        if (operation == "var") {
            int rows, columns;
            cin >> rows >> columns;
            vector<int> parents = {};
            inputs.emplace_back(rows, columns);
            Var *var = new Var(parents);
            graph.add(var);
        } else if (operation == "tnh") {
            int parent;
            cin >> parent;
            vector<int> parents = {parent - 1};
            Tanh *tanh = new Tanh(parents);
            graph.add(tanh);
        } else if (operation == "rlu") {
            double alpha;
            cin >> alpha;
            int parent;
            cin >> parent;
            vector<int> parents = {parent - 1};
            ReLu *reLu = new ReLu(parents, alpha);
            graph.add(reLu);
        } else if (operation == "mul") {
            int first, second;
            cin >> first >> second;
            vector<int> parents = {first - 1, second - 1};
            Mul *mul = new Mul(parents);
            graph.add(mul);
        } else if (operation == "sum") {
            int number;
            cin >> number;
            vector<int> parents;
            for (int j = 0; j < number; j++) {
                int parent;
                cin >> parent;
                parents.push_back(parent - 1);
            }
            Sum *sum = new Sum(parents);
            graph.add(sum);
        } else if (operation == "had") {
            int number;
            cin >> number;
            vector<int> parents;
            for (int j = 0; j < number; j++) {
                int parent;
                cin >> parent;
                parents.push_back(parent - 1);
            }
            Had *had = new Had(parents);
            graph.add(had);
        } else {
            cout << "Насрал" << "\n";
        }
    }
    for (int i = 0; i < m; i++) {
        int rows = inputs[i].first;
        int columns = inputs[i].second;
        vector<vector<double>> values(rows, vector<double>(columns, 0));
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                cin >> values[row][column];
            }
        }
        graph[i].matrix = {rows, columns, values};
    }
    calculate(graph, k);
    for (int i = 0; i < n; i++) {
        graph[i].init_gradient();
    }
    int size = (int) graph.nodes.size();
    for (int i = size - k; i < size; i++) {
        int rows = graph[i].matrix.rows;
        int columns = graph[i].matrix.columns;
        vector<vector<double>> values(rows, vector<double>(columns, 0));
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                cin >> values[row][column];
            }
        }
        Matrix plus = {rows, columns, values};
        graph[i].gradient = graph[i].gradient + plus;
    }
    derivative(graph, m);
}
