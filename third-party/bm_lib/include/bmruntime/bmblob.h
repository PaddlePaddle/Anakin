#ifndef __BM_BLOB_H__
#define __BM_BLOB_H__

struct bm_mem_desc;
typedef struct bm_mem_desc bm_device_mem_t;
namespace bmcnn {

typedef struct { int n, c, h, w; } Shape;

class BMBlob
{
public:
    /**
     * \brief Constructor of blob.
     *
     * \param shape - Shape of blob
     */
    explicit BMBlob(const Shape &shape, void *handle);
    /**
     * \brief Deconstructor of blob.
     */
    virtual ~BMBlob();
    /**
     * \brief Reshape blob.
     * 
     * \param n - Batch number of blob
     * \param c - Channel number of blob
     * \param h - Height of blob section
     * \param w - Width of blob section
     *
     * \note
     * (1) For now, number of channels is not allowed to be reshaped.\n
     * (2) After reshaping, data in this blob will be set vanished.\n
     */
    void Reshape(int n, int c, int h, int w);
    /**
     * \brief Get shape.
     */
    inline Shape shape() const
    { return shape_; }
    /**
     * \brief Get batch size.
     */
    inline int batch_num() const
    { return shape_.n; }
    /**
     * \brief Get feature
     *
     * \return Channel number of the blob\n
     */
    inline int channels() const
    { return shape_.c; }
    /**
     * \brief Get height of section
     */
    int height() const
    { return shape_.h; }
    /**
     * \brief Get width of section.
     */
    int width() const
    { return shape_.w; }
    /**
     * \brief Get read-only pointer to data in cpu.
     */
    const float *cpu_data(); 
    /**
     * \brief Get mutable pointer of data in cpu.
     */    
    float *mutable_cpu_data();
    /**
     * \brief Get mutable pointer of memory in device.
     */    
    bm_device_mem_t *mutable_dev_mem();
    /**
     * \brief Get read-only pointer of memory in device.
     */    
    const bm_device_mem_t *dev_mem();
private:
    BMBlob(const BMBlob &other);
    BMBlob &operator=(const BMBlob &other);
    
    bm_device_mem_t *dev_mem_;
    float *sys_data_;
    Shape shape_;
    int data_pos_;
    int capacity_;
    void *handle_;
    
    enum { AIR = 0x00, SYS = 0x01, DEV = 0x10 };
    void sync_s2d();
    void sync_d2s();
};

} /* namespace bmcnn */

#endif /* __BM_BLOB_H__ */
