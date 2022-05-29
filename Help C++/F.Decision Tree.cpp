#include <bits/stdc++.h>

using namespace std;

struct Criteria {
    Criteria() {
        index = 0;
        criteria = 0;
    }

    explicit Criteria(int index, float criteria) {
        this->index = index;
        this->criteria = criteria;
    }

    ostream &print(ostream &ostream) const {
        ostream << index << " " << criteria;
        return ostream;
    }

    int index;
    float criteria;
};

struct Vertex {
    virtual ostream &print(ostream &ostream) = 0;
};

struct Node : Vertex {
    Node(int left, int right, Criteria &criteria) {
        this->left = left;
        this->right = right;
        this->criteria = criteria;
    }

    ostream &print(ostream &ostream) override {
        ostream << "Q" << " ";
        criteria.print(ostream) << " " << left << " " << right;
        return ostream;
    }

private:
    int left;
    int right;
    Criteria criteria;
};

struct Leaf : Vertex {
    explicit Leaf(int number) {
        this->number = number;
    }

    ostream &print(ostream &ostream) override {
        ostream << "C" << " " << number;
        return ostream;
    }

private:
    int number;
};

struct DecisionTree {
    explicit DecisionTree(function<float(vector<float> &, float)> &functional, int height) {
        this->functional = functional;
        this->height = height;
        vertexes = vector<Vertex *>(1 << (height + 1));
    }

    void fit(vector<pair<vector<float>, int>> &train, int features, int classes) {
        int size = split(train, features, classes, 1, 1, 0, (int) train.size());
        vector<Vertex *> result(size);
        for (int i = 0; i < size; i++) {
            result[i] = vertexes[i];
        }
        vertexes = result;
    }

    friend ostream &operator<<(ostream &ostream, DecisionTree &tree);

private:
    float branch_criteria(vector<float> &parent, float all,
                                vector<float> &left, float left_size,
                                vector<float> &right, float right_size) {
        float result = functional(parent, all);
        result -= (left_size / all) * functional(left, left_size);
        result -= (right_size / all) * functional(right, right_size);
        return result;
    }

    int split(vector<pair<vector<float>, int>> &train, int features, int classes, int current, int number, int start,
              int end) {
        // Leaf
        int answer = 0;
        vector<float> counts(classes, 0);
        for (int i = start; i < end; i++) {
            auto element = train[i];
            counts[element.second - 1]++;
            if (answer == 0 || answer == element.second) {
                answer = element.second;
            } else {
                answer = -1;
            }
        }
        if (current > height) {
            float count = 0;
            for (int i = 0; i < classes; i++) {
                if (counts[i] > count) {
                    answer = i + 1;
                    count = counts[i];
                }
            }
        }
        if (answer != -1) {
            vertexes[number - 1] = new Leaf(answer);
            return 1;
        }
        // Node
        int best_index = -1;
        float best_criteria = 0;
        float best_score = 0;
        for (int i = 0; i < features; i++) {
            vector<float> left(classes, 0);
            vector<float> right(counts);
            sort(train.begin() + start, train.begin() + end,
                 [i](pair<vector<float>, int> &first, pair<vector<float>, int> &second) {
                     return first.first[i] < second.first[i];
                 });
            int index = start;
            while (index < end - 1) {
                float value = train[index].first[i];
                if (train[end - 1].first[i] == value) {
                    break;
                }
                while (value == train[index].first[i]) {
                    left[train[index].second - 1]++;
                    right[train[index].second - 1]--;
                    index++;
                }
                float score = branch_criteria(counts, (float) (end - start),
                                                    left, (float) (index - start),
                                                    right, (float) (end - index));
                if (score > best_score) {
                    best_score = score;
                    best_index = i;
                    best_criteria = (train[index - 1].first[i] + train[index].first[i]) / 2;
                }
            }
        }
        sort(train.begin() + start, train.begin() + end,
             [best_index](pair<vector<float>, int> &first, pair<vector<float>, int> &second) {
                 return first.first[best_index] < second.first[best_index];
             });
        int index = start;
        while (train[index].first[best_index] < best_criteria) {
            index++;
        }
        int left = split(train, features, classes, current + 1, number + 1, start, index);
        int right = split(train, features, classes, current + 1, number + left + 1, index,
                          end);
        Criteria best = Criteria(best_index + 1, best_criteria);
        vertexes[number - 1] = new Node(number + 1, number + left + 1, best);
        return 1 + left + right;
    }

    function<float(vector<float> &, float)> functional;
    int height;
    vector<Vertex *> vertexes;
};

ostream &operator<<(ostream &ostream, DecisionTree &tree) {
    ostream << tree.vertexes.size() << "\n";
    for (Vertex *vertex: tree.vertexes) {
        vertex->print(ostream);
        ostream << "\n";
    }
    return ostream;
}

float entropy(vector<float> &classes, float number) {
    float result = 0;
    for (float &i: classes) {
        if (i != 0) {
            float probability = i / number;
            result -= probability * log(probability);
        }
    }
    return result;
}

float gini(vector<float> &classes, float number) {
    float result = 1;
    for (float &i: classes) {
        if (i != 0) {
            float probability = i / number;
            result -= probability * probability;
        }
    }
    return result;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    int features_number, classes, height, number;
    cin >> features_number >> classes >> height >> number;
    function<float(vector<float> &, float)> functional = number > 100 ? gini : entropy;
    DecisionTree tree = DecisionTree(functional, height);
    vector<pair<vector<float>, int>> train;
    for (int i = 0; i < number; i++) {
        vector<float> features;
        for (int j = 0; j < features_number; j++) {
            float feature;
            cin >> feature;
            features.push_back(feature);
        }
        int index;
        cin >> index;
        train.emplace_back(features, index);
    }
    tree.fit(train, features_number, classes);
    cout << tree;
}
