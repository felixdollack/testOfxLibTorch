#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    torch::manual_seed(1);

    // check if we can use GPU
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        this->device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        this->device_type = torch::kCPU;
    }

    torch::Device device(this->device_type);
    this->model = new Net();
    this->model->to(device);
    this->modelIsTrained = false;
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key == '1') {
        trainMNIST();
        this->modelIsTrained = true;
    }
    if (key == 's') {
        if (this->modelIsTrained) {
            // changes serialize.h to make this work
            string model_path = ofToDataPath("net.pt");
            torch::serialize::OutputArchive output_archive;
            this->model->save(output_archive);
            output_archive.save_to(model_path);
            //torch::save(this->model, ofToDataPath("net.pt"));
            cout << "Saved network." << endl;
        } else {
            cout << "Network is not ready for saving yet." << endl;
        }
    }
    if (key == 'l') {
        try {
            // Load the model
            torch::serialize::InputArchive archive;
            std::string file(ofToDataPath("net.pt"));
            archive.load_from(file);
            this->model->load(archive);
            //torch::load(this->model, ofToDataPath("net.pt"));
            this->modelIsTrained = true;
            cout << "Loaded network." << endl;
        } catch (const c10::Error& e) {
            cerr << "Error loading the model" << endl;
        }
    }
    if (key == 't') {
        if (modelIsTrained) {
            testMNIST();
        } else {
            cout << "Network is not ready testing yet." << endl;
        }
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}

void ofApp::trainMNIST()
{
    torch::Device device(this->device_type);

    /* prepare training data loader */
    auto train_dataset = torch::data::datasets::MNIST(ofToDataPath(kDataRoot))
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    auto train_loader =
    torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), kTrainBatchSize);

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        // train
        // train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        model->train();
        size_t batch_idx = 0;
        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device),
            targets = batch.target.to(device);

            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            auto output = model->forward(data);
            // Compute a loss value to judge the prediction of our model.
            auto loss = torch::nll_loss(output, targets);
            AT_ASSERT(!std::isnan(loss.template item<float>()));
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();

            if (batch_idx++ % kLogInterval == 0) {
                std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
                            epoch, batch_idx * batch.data.size(0),
                            train_dataset_size, loss.template item<float>());
            }
            // Output the loss every 10 batches.
            if (++batch_idx % kLogInterval == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_idx << " | Loss: " << loss.item<float>() << std::endl;
            }
        }
    }
}

void ofApp::testMNIST() {
    torch::Device device(this->device_type);
    /* prepare test data loader */
    auto test_dataset = torch::data::datasets::MNIST(ofToDataPath(kDataRoot), torch::data::datasets::MNIST::Mode::kTest)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader =
    torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

    torch::NoGradGuard no_grad;
    model->eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (const auto& batch : *test_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        auto output = model->forward(data);
        test_loss += torch::nll_loss(output, targets,
                                     /*weight=*/{}, Reduction::Sum)
        .template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }

    test_loss /= test_dataset_size;
    std::printf("\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
                test_loss, static_cast<double>(correct) / test_dataset_size);
}
