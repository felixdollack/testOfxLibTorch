#pragma once

#include "ofMain.h"
#include "ofxLibTorch.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

    /* code adapted from https://github.com/pytorch/examples/blob/master/cpp/mnist/mnist.cpp */

    // Where to find the MNIST dataset.
    const char* kDataRoot = "mnist";
    // The batch size for training.
    const int64_t kTrainBatchSize = 64;
    // The batch size for testing.
    const int64_t kTestBatchSize = 1000;
    // The number of epochs to train.
    const int64_t kNumberOfEpochs = 10;
    // After how many batches to log a new update with the loss value.
    const int64_t kLogInterval = 10;
    // After how many batches to save a checkpoint of the model.
    const int64_t kSaveInterval = 100;

    // Define a new Module.
    struct Net : torch::nn::Module {
        Net()
        : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10) {
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("conv2_drop", conv2_drop);
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
            x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
            x = x.view({-1, 320});
            x = torch::relu(fc1->forward(x));
            x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
            x = fc2->forward(x);
            return torch::log_softmax(x, /*dim=*/1);
        }

        torch::nn::Conv2d conv1;
        torch::nn::Conv2d conv2;
        torch::nn::FeatureDropout conv2_drop;
        torch::nn::Linear fc1;
        torch::nn::Linear fc2;
    };

    Net* model;
    torch::DeviceType device_type;
    bool modelIsTrained;
    void trainMNIST();
    void testMNIST();

    string testResult;
};
