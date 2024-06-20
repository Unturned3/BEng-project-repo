
#pragma once

#include <map>
#include <utility>
#include <vector>
#include <tuple>

#include <H5Cpp.h>
#include <H5PredType.h>
#include <Eigen/Dense>

#include "fmt/format.h"

#include "ImagePair.hpp"
#include "Track.hpp"

namespace MyH5 {

std::vector<hsize_t> getDataSetDims(H5::DataSet& ds)
{
    auto ndims = ds.getSpace().getSimpleExtentNdims();
    std::vector<hsize_t> dims(ndims);
    ds.getSpace().getSimpleExtentDims(dims.data(), nullptr);
    return dims;
}

}  // namespace MyH5

std::pair<std::vector<int>, std::vector<ImagePair>> load_image_pairs(
    const std::string& path)
{
    using namespace H5;

    std::vector<ImagePair> pairs;

    H5File file(path, H5F_ACC_RDONLY);

    int n_pairs;
    file.openAttribute("n_pairs").read(PredType::NATIVE_INT, &n_pairs);

    for (int i = 0; i < n_pairs; i++) {
        ImagePair p;
        Group g = file.openGroup(fmt::format("pair_{}", i));

        // Read H
        DataSet H_dataset = g.openDataSet("H");
        H_dataset.read(p.H, PredType::NATIVE_DOUBLE);

        // Read src_pts
        DataSet src_pts_dataset = g.openDataSet("src_pts");
        hsize_t src_pts_dims[2];
        src_pts_dataset.getSpace().getSimpleExtentDims(src_pts_dims, NULL);
        p.src_pts.resize(src_pts_dims[0]);
        /*  NOTE: This is a bit fishy; we're assuming that a vector of arrays
            is stored contiguously in memory. It seems to work though. */
        src_pts_dataset.read(p.src_pts.data()->data(), PredType::NATIVE_DOUBLE);

        // Read dst_pts
        DataSet dst_pts_dataset = g.openDataSet("dst_pts");
        hsize_t dst_pts_dims[2];
        dst_pts_dataset.getSpace().getSimpleExtentDims(dst_pts_dims, NULL);
        p.dst_pts.resize(dst_pts_dims[0]);
        dst_pts_dataset.read(p.dst_pts.data()->data(), PredType::NATIVE_DOUBLE);

        // Read attributes i and j
        g.openAttribute("i").read(PredType::NATIVE_INT, &p.i);
        g.openAttribute("j").read(PredType::NATIVE_INT, &p.j);
        g.openAttribute("still").read(PredType::NATIVE_HBOOL, &p.still);

        pairs.push_back(p);
    }

    std::vector<int> cam_indices;
    DataSet cam_indices_dataset = file.openDataSet("cam_indices");
    hsize_t cam_indices_dims[2];
    cam_indices_dataset.getSpace().getSimpleExtentDims(cam_indices_dims, NULL);
    cam_indices.resize(cam_indices_dims[0]);
    cam_indices_dataset.read(cam_indices.data(), PredType::NATIVE_INT);
    return {cam_indices, pairs};
}

std::map<int, std::array<double, 4>> import_cam_params(const std::string& path)
{
    using namespace H5;

    H5File file(path, H5F_ACC_RDONLY);
    std::map<int, std::array<double, 4>> cam_params;

    fmt::print("Num objects in file: {}\n", file.getNumObjs());

    for (int idx = 0; idx < (int)file.getNumObjs(); idx++) {
        auto name = file.getObjnameByIdx(idx);
        if (file.childObjType(name) != H5O_TYPE_DATASET)
            continue;
        DataSet ds = file.openDataSet(name);
        auto dims = MyH5::getDataSetDims(ds);
        int cam_idx;
        std::array<double, 4> params;
        ds.openAttribute("cam_idx").read(PredType::NATIVE_INT, &cam_idx);
        ds.read(params.data(), PredType::NATIVE_DOUBLE);
        cam_params[cam_idx] = params;
    }

    return cam_params;
}

void export_cam_params(const std::string& path,
                       std::map<int, std::array<double, 4>> cam_params)
{
    using namespace H5;

    H5File file(path, H5F_ACC_TRUNC);

    for (const auto& [cam_idx, params] : cam_params) {
        std::string name = fmt::format("cam_{}", cam_idx);

        hsize_t dims[1] = {4};
        DataSpace dataspace(1, dims);

        DataSet dataset =
            file.createDataSet(name, PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(params.data(), PredType::NATIVE_DOUBLE);

        dataset
            .createAttribute("cam_idx", PredType::NATIVE_INT,
                             DataSpace(H5S_SCALAR))
            .write(PredType::NATIVE_INT, &cam_idx);
    }
}

std::tuple<int, int, std::vector<std::vector<Eigen::Vector2d>>>
load_sift_tracks(const std::string& path)
{
    using namespace H5;
    H5File f(path, H5F_ACC_RDONLY);
    std::vector<std::vector<Eigen::Vector2d>> tracks;

    int f_begin;
    f.openAttribute("f_begin").read(PredType::NATIVE_INT, &f_begin);

    auto g = f.openGroup(fmt::format("keyframe_{}", f_begin));

    int n_pts;
    g.openAttribute("n_pts").read(PredType::NATIVE_INT, &n_pts);

    fmt::print("f_begin: {}\n", f_begin);
    fmt::print("n_pts: {}\n", n_pts);

    tracks.resize(n_pts);

    for (int i = 0; i < n_pts; i++) {
        auto ds = g.openDataSet(fmt::format("pt_{}", i));
        auto dims = MyH5::getDataSetDims(ds);
        tracks[i].resize(dims[0]);
        ds.read(tracks[i][0].data(), PredType::NATIVE_DOUBLE);
    }

    return {f_begin, f_begin + tracks[0].size(), tracks};
}

std::pair<std::vector<int>, std::vector<Track>> load_tracks(
    const std::string& path)
{
    using namespace H5;

    std::vector<int> track_uids;
    std::vector<Track> tracks;

    H5File file(path, H5F_ACC_RDONLY);

    int n_tracks;
    file.openAttribute("n_tracks").read(PredType::NATIVE_INT, &n_tracks);

    DataSet track_uids_dataset = file.openDataSet("track_uids");
    track_uids.resize(static_cast<size_t>(n_tracks));
    track_uids_dataset.read(track_uids.data(), PredType::NATIVE_INT);

    for (int uid : track_uids) {
        Track t;
        t.uid = uid;
        Group g = file.openGroup(fmt::format("track_{}", uid));

        DataSet pts_dataset = g.openDataSet("pts");
        hsize_t src_pts_dims[2];
        pts_dataset.getSpace().getSimpleExtentDims(src_pts_dims, NULL);
        t.pts.resize(src_pts_dims[0]);
        pts_dataset.read(t.pts.data(), PredType::NATIVE_DOUBLE);

        g.openAttribute("start_frame_idx")
            .read(PredType::NATIVE_INT, &t.start_frame_idx);
        tracks.push_back(t);
    }

    return {track_uids, tracks};
}
