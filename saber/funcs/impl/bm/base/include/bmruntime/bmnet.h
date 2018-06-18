#ifndef __BM_NET_H__
#define __BM_NET_H__

#include "bmblob.h"
#include "bmcnnctx.h"
#include <vector>
#include <map>
#include <string>

#ifdef CROSS_COMPILE
  #include <memory>
#else
  #include <boost/shared_ptr.hpp>
#endif


#ifdef CROSS_COMPILE
#define NAMESPACE_USED  std
#else
#define NAMESPACE_USED  boost
#endif

namespace bmcnn {
    
class BMNet
{
public:
    /**
     * \brief Constructor of net.
     *
     * \param handle - Handler of BMCNN context (created by \ref bmcnn_ctx_create)
     * \param name - Name of net
     */
    explicit BMNet(bmcnn_ctx_t handle, const std::string &name);
    /**
     * \brief Deconstructor of blob.
     */
    virtual ~BMNet();
    /**
     * \brief Reshape all layers from bottom to top.
     */
    void Reshape();
    /**
     * \brief Run forward.
     * 
     * \param sync - Flag of synchronizing.
     */
    void Forward(bool sync = false);
    /**
     * \brief Get blob by name.
     *
     * \param name - Name of blob 
     * \note
     * (1) The name could only be of blob in input or output.\n
     * (2) If the name is not spotted, null pointer will be returned.\n
     */
    const NAMESPACE_USED::shared_ptr<BMBlob> blob_by_name(const std::string &name) const;
    /**
     * \brief Get maximum shape allowed.
     */
    inline const Shape &max_shape() const
    { return max_shape_; }
private:
    BMNet(const BMNet &other);
    BMNet &operator=(const BMNet &other);

    bmcnn_ctx_t bmcc_ctx_;
    std::vector<NAMESPACE_USED::shared_ptr<BMBlob> > blobs_;
    std::vector<BMBlob *> net_input_blobs_;
    std::vector<BMBlob *> net_output_blobs_;
    std::string name_;
    std::map<std::string, size_t> blob_name_index_;
    Shape max_shape_;
};

} /* namespace bmcnn */

#endif /* __BM_NET_H__ */
