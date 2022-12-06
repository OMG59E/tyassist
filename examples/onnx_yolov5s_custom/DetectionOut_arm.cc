#include <string>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>


typedef struct {
    int x1{0};
    int y1{0};
    int x2{0};
    int y2{0};
} box_t;


typedef struct {
    float conf{0.0f};
    int cls{-1};  // cls index
    box_t box;
} detection_t;


inline float bbox_overlap(const box_t &vi, const box_t &vo) {
    int xx1 = std::max(vi.x1, vo.x1);
    int yy1 = std::max(vi.y1, vo.y1);
    int xx2 = std::min(vi.x2, vo.x2);
    int yy2 = std::min(vi.y2, vo.y2);

    int w = std::max(0, xx2 - xx1);
    int h = std::max(0, yy2 - yy1);

    int area = w * h;

    float dist = float(area) / float((vi.x2 - vi.x1) * (vi.y2 - vi.y1) +
                                     (vo.y2 - vo.y1) * (vo.x2 - vo.x1) - area);

    return dist;
}


static int non_max_suppression(std::vector<detection_t> &detections, const float iou_threshold) {
    // sort
    std::sort(detections.begin(), detections.end(),
              [](const detection_t &d1, const detection_t &d2) { return d1.conf > d2.conf; });

    // nms
    std::vector<detection_t> keep_detections;
    std::vector<detection_t> tmp_detections;
    keep_detections.clear();
    while (!detections.empty()) {
        if (detections.size() == 1) {
            keep_detections.emplace_back(detections[0]);
            break;
        }

        keep_detections.emplace_back(detections[0]);

        tmp_detections.clear();
        for (int idx = 1; idx < detections.size(); ++idx) {
            float iou = bbox_overlap(keep_detections.back().box, detections[idx].box);
            if (iou < iou_threshold)
                tmp_detections.emplace_back(detections[idx]);
        }
        detections.swap(tmp_detections);
    }
    detections.swap(keep_detections);
    return 0;
}

extern "C" int
DetectionOut(int input_num, int output_num, void *input, int *input_shape, char *input_dtype, void *output,
             int *output_shape, char *output_dtype, float conf_threshold, float iou_threshold, int top_k,
             int keep_top_k, int num_classes) {

    // Write your code below
    assert(input_num == 1);
    assert(output_num == 1);
    assert(strcmp(input_dtype, "float32") == 0);
    assert(strcmp(output_dtype, "float32") == 0);
    assert(3 == input_shape[0]);
    assert(3 == output_shape[0]);
    assert(1 == input_shape[1]); // only bs1
    assert(1 == output_shape[1]); // only bs1
    assert(keep_top_k == output_shape[2]);
    assert(6 == output_shape[3]);

    memset(output, 0, 1 * keep_top_k * 6 * sizeof(float));

    const int min_wh = 2;
    const int max_wh = 7680;

    const float *input_data = reinterpret_cast<const float *>(input);

    std::vector<detection_t> detections;
    detections.clear();

    const int num_anchors = input_shape[2];
    const int step = num_classes + 5;
    assert(step == input_shape[3]);

    for (int dn = 0; dn < num_anchors; ++dn) {
        float conf = input_data[dn * step + 4];
        if (conf < conf_threshold)
            continue;

        float w = input_data[dn * step + 2];
        float h = input_data[dn * step + 3];

        if (w < min_wh || h < min_wh || w > max_wh || h > max_wh)
            continue;

        float cx = input_data[dn * step + 0];
        float cy = input_data[dn * step + 1];

        detection_t detection;
        detection.box.x1 = cx - w * 0.5f;
        detection.box.y1 = cy - h * 0.5f;
        detection.box.x2 = cx + w * 0.5f;
        detection.box.y2 = cy + h * 0.5f;
        int num_cls{-1};
        float max_conf{-1};
        for (int dc = 0; dc < num_classes; ++dc) {  // [0-80)
            float score = input_data[dn * step + 5 + dc] * conf;
            if (max_conf < score) {
                num_cls = dc;
                max_conf = score;
            }
        }
        detection.cls = num_cls;
        detection.conf = max_conf;
        detections.emplace_back(detection);
    }

    if (detections.empty())
        return 1;

    // nms
    non_max_suppression(detections, iou_threshold);

    float *output_data = reinterpret_cast<float *>(output);
    for (int dn = 0; dn < detections.size(); ++dn) {
        if (dn >= keep_top_k)
            break;
        output_data[dn * 6 + 0] = detections[dn].box.x1;
        output_data[dn * 6 + 1] = detections[dn].box.y1;
        output_data[dn * 6 + 2] = detections[dn].box.x2;
        output_data[dn * 6 + 3] = detections[dn].box.y2;
        output_data[dn * 6 + 4] = detections[dn].conf;
        output_data[dn * 6 + 5] = detections[dn].cls;
    }

    return 1;
}