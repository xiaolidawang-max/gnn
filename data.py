from torch_geometric.data import Data


# Override "Data" class to ensure correct batching for pooling hierarchy
class MultiscaleData(Data):

    def __inc__(self, key, value, *args, **kwargs):
        if "batch" in key:
            return int(value.max()) + 1

        # Batch scales correctly
        elif "scale" in key and (
            "sample_index" in key or "pool_source" in key or "interp_target" in key
        ):
            if int(key[5]) == 0:
                return self.num_nodes
            else:
                return self["scale" + str(int(key[5]) - 1) + "_sample_index"].size(
                    dim=0
                )  # must pass integer (PyTorch)

        elif "scale" in key and ("pool_target" in key or "interp_source" in key):
            return self["scale" + key[5] + "_sample_index"].size(dim=0)

        # Batch edges and polygons as before
        elif "_index" in key or "face" in key or key == "tets" or "wcm_" in key:
            return self.num_nodes

        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if "edge_index" in key or "face" in key or key == "tets":
            return -1
        else:
            return 0
