#include <gpd/net/classifier.h>
#if defined(USE_OPENVINO)
#include <gpd/net/openvino_classifier.h>
#elif defined(USE_LIBTORCH)
#include <gpd/net/libtorch_classifier.h>
#elif defined(USE_PYTHON)
#include <gpd/net/python_classifier.h>
#else
#include <gpd/net/eigen_classifier.h>
#endif

namespace gpd {
namespace net {

std::shared_ptr<Classifier> Classifier::create(const std::string &model_file, const std::string &weights_file,
                                               Classifier::Device device, int batch_size) {
#if defined(USE_OPENVINO)
  return std::make_shared<OpenVinoClassifier>(model_file, weights_file, device, batch_size);
#elif defined(USE_LIBTORCH)
  return std::make_shared<LibtorchClassifier>(model_file, weights_file, device, batch_size);
#elif defined(USE_PYTHON)
  return std::make_shared<PythonClassifier>(model_file, weights_file, device, batch_size);
#else
  return std::make_shared<EigenClassifier>(model_file, weights_file, device, batch_size);
#endif
}

}  // namespace net
}  // namespace gpd
