import os
from typing import Optional, Tuple, List, Dict

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from torchtyping import TensorType#, patch_typeguard
from typeguard import typechecked

from gb.typing import Int, Float
from gb.util import FALLBACK_SRC_PATH

# patch_typeguard()

Dataset = Tuple[
    TensorType["nodes", "nodes", torch.float32, torch.strided],
    TensorType["nodes", "features", torch.float32, torch.strided],
    Optional[TensorType["nodes", torch.int64, torch.strided]]
]


@typechecked
def get_dataset(dataset_name: str) -> Dataset:
    if dataset_name in ("citeseer", "cora"):
        try:
            return _load_npz(os.path.join(os.path.dirname(__file__), "..", "data", dataset_name + ".npz"))
        except FileNotFoundError:
            # Fallback for runs via SEML on the GPU cluster.
            return _load_npz(f"{FALLBACK_SRC_PATH}/data/{dataset_name}.npz")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


@typechecked
def _load_npz(path: str) -> Dataset:
    with np.load(path, allow_pickle=True) as loader:
        loader = dict(loader)
        A = _fix_adj_mat(_extract_csr(loader, "adj"))
        _, comp_ids = connected_components(A)
        lcc_nodes = np.nonzero(comp_ids == mode(comp_ids)[0])[0]
        A = torch.tensor(A[lcc_nodes, :][:, lcc_nodes].todense(), dtype=torch.float32)
        if "attr_data" in loader:
            X = torch.tensor(_extract_csr(loader, "attr")[lcc_nodes, :].todense(), dtype=torch.float32)
        else:
            X = torch.eye(A.shape[0])
        if "labels" in loader:
            y = torch.tensor(loader["labels"][lcc_nodes], dtype=torch.int64)
        else:
            y = None
        return A, X, y


@typechecked
def _extract_csr(loader, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (loader[f"{prefix}_data"], loader[f"{prefix}_indices"], loader[f"{prefix}_indptr"]),
        loader[f"{prefix}_shape"]
    )


@typechecked
def _fix_adj_mat(A: sp.csr_matrix) -> sp.csr_matrix:
    # Some adjacency matrices do have some values on the diagonal, but not everywhere. Get rid of this mess.
    A = A - sp.diags(A.diagonal())
    # For some reason, some adjacency matrices are not symmetric. Fix this following the Nettack code.
    A = A + A.T
    A[A > 1] = 1
    return A


@typechecked
def get_splits(y: TensorType[-1]) \
        -> List[Tuple[TensorType[-1, torch.int64], TensorType[-1, torch.int64], TensorType[-1, torch.int64]]]:
    """
    Produces 5 deterministic 10-10-80 splits.
    """
    return [_three_split(y, 0.1, 0.1, random_state=r) for r in [1234, 2021, 1309, 4242, 1998]]


@typechecked
def _three_split(y: TensorType[-1], size_1: Float, size_2: Float, random_state: Int) \
        -> Tuple[TensorType[-1, torch.int64], TensorType[-1, torch.int64], TensorType[-1, torch.int64]]:
    idx = np.arange(y.shape[0])
    idx_12, idx_3 = train_test_split(idx, train_size=size_1 + size_2, stratify=y, random_state=random_state)
    idx_1, idx_2 = train_test_split(idx_12, train_size=size_1 / (size_1 + size_2), stratify=y[idx_12],
                                    random_state=random_state)
    return torch.tensor(idx_1), torch.tensor(idx_2), torch.tensor(idx_3)


@typechecked
def get_all_benchmark_targets() -> List[str]:
    return ["degree_1", "degree_2", "degree_3", "degree_5", "degree_8_to_10", "degree_15_to_25"]


@typechecked
def get_num_nodes_per_benchmark_target() -> int:
    return 20


@typechecked
def get_benchmark_target_nodes(dataset_name: str) -> Dict[str, TensorType[5, 20, torch.int64, torch.strided]]:
    """
    These benchmarks have been generated by the make_benchmark notebook.
    """
    # @formatter:off
    if dataset_name == "citeseer":
        return {
            "degree_1": torch.tensor([[  499,   698,  1011,  1012,  1087,  1101,  1125,  1135,  1331,  1411,  1532,  1618,  1646,  1732,  1810,  1864,  1887,  1902,  1935,  1936],
                                      [   12,   120,   301,   416,   482,   540,   633,   659,  1101,  1228,  1416,  1598,  1664,  1745,  1758,  1835,  1935,  1943,  1964,  1994],
                                      [   51,   182,   294,   334,   365,   427,   473,   528,   549,   561,   608,   693,   861,  1090,  1096,  1354,  1593,  1651,  1704,  1709],
                                      [   44,   229,   365,   499,   549,   608,   731,   789,   872,  1061,  1104,  1347,  1501,  1546,  1622,  1920,  1956,  2005,  2057,  2074],
                                      [  153,   269,   563,   731,   744,   945,   972,   997,  1085,  1215,  1217,  1241,  1354,  1382,  1589,  1623,  1772,  1792,  1889,  2103]]),
            "degree_2": torch.tensor([[  148,   293,   510,   587,   751,   772,   773,   776,   923,  1168,  1237,  1296,  1358,  1431,  1712,  1730,  1823,  1888,  1953,  2049],
                                      [  159,   243,   806,   825,   907,   921,   932,  1017,  1119,  1165,  1353,  1372,  1376,  1422,  1500,  1522,  1672,  1763,  1850,  1922],
                                      [  222,   226,   248,   328,   336,   518,   924,   993,  1161,  1168,  1275,  1284,  1502,  1586,  1628,  1730,  1735,  1769,  1895,  2017],
                                      [   57,   135,   159,   284,   342,   533,   750,   806,   931,   966,   973,  1078,  1305,  1435,  1502,  1586,  1662,  1754,  1823,  1850],
                                      [  226,   428,   562,   647,   672,   776,   816,   828,   924,   986,  1113,  1127,  1207,  1284,  1522,  1620,  1807,  1816,  1897,  2102]]),
            "degree_3": torch.tensor([[    2,   251,   355,   402,   606,   813,   835,   846,   878,   964,  1014,  1081,  1190,  1503,  1519,  1578,  1818,  1969,  2031,  2043],
                                      [   67,   188,   278,   346,   356,   414,   468,   535,   543,   585,  1040,  1261,  1344,  1503,  1741,  1789,  1824,  1913,  2000,  2030],
                                      [    1,   360,   368,   379,   400,   535,   589,   859,   990,  1057,  1170,  1282,  1298,  1406,  1471,  1690,  1863,  1958,  1993,  2080],
                                      [   21,    87,   170,   278,   345,   468,   835,   929,   964,  1059,  1138,  1143,  1216,  1234,  1261,  1332,  1520,  1789,  1913,  2001],
                                      [   21,    41,    72,   188,   231,   464,   553,   625,   925,  1173,  1181,  1212,  1261,  1567,  1666,  1705,  1813,  1997,  2058,  2073]]),
            "degree_5": torch.tensor([[   34,   209,   297,   327,   364,   576,   732,   756,   858,   919,  1075,  1193,  1227,  1242,  1243,  1263,  1711,  1767,  1875,  1908],
                                      [  105,   155,   162,   327,   341,   451,   472,   652,   853,   857,   858,   919,  1151,  1156,  1440,  1446,  1516,  1612,  2038,  2098],
                                      [   36,   273,   364,   451,   458,   576,   652,   974,  1075,  1227,  1242,  1440,  1509,  1631,  1657,  1711,  1767,  1908,  2053,  2089],
                                      [  105,   155,   273,   327,   350,   558,   576,   650,   652,   664,   853,   875,   977,  1156,  1223,  1299,  1363,  1733,  1981,  2061],
                                      [   34,    36,    82,   327,   367,   497,   704,   735,   756,   854,   916,  1223,  1385,  1440,  1509,  1631,  1833,  2038,  2061,  2107]]),
            "degree_8_to_10": torch.tensor([[   63,   109,   211,   420,   486,   524,   614,   622,   645,   860,   898,   908,  1049,  1144,  1246,  1338,  1468,  1551,  1671,  1801],
                                            [   63,   109,   156,   225,   519,   622,   684,   740,   823,   888,   950,  1069,  1196,  1246,  1252,  1671,  1917,  2015,  2056,  2083],
                                            [   83,   109,   519,   786,   860,   908,   949,   999,  1152,  1246,  1355,  1379,  1423,  1671,  1770,  1790,  1801,  1843,  1917,  2068],
                                            [   63,    98,   114,   122,   393,   524,   595,   860,   949,  1195,  1246,  1285,  1355,  1370,  1423,  1468,  1551,  1862,  1972,  1988],
                                            [   83,    98,   207,   486,   524,   602,   662,   885,   950,  1069,  1152,  1285,  1338,  1423,  1543,  1552,  1671,  1770,  1972,  2068]]),
            "degree_15_to_25": torch.tensor([[   85,   144,   598,   688,   904,  1021,  1027,  1079,  1311,  1333,  1388,  1390,  1410,  1453,  1481,  1635,  1687,  1694,  2062,  2084],
                                             [   85,   144,   268,   354,   598,  1027,  1079,  1289,  1311,  1333,  1388,  1410,  1453,  1523,  1687,  1706,  1894,  2062,  2084,  2090],
                                             [  144,   354,   904,  1021,  1027,  1067,  1079,  1289,  1333,  1348,  1388,  1390,  1453,  1523,  1554,  1635,  1687,  1706,  1894,  2090],
                                             [   85,   354,   598,   688,  1027,  1067,  1079,  1289,  1311,  1333,  1348,  1390,  1481,  1554,  1635,  1687,  1694,  1706,  2062,  2090],
                                             [  144,   268,   598,   688,  1021,  1027,  1067,  1079,  1289,  1333,  1348,  1390,  1453,  1481,  1523,  1635,  1687,  1894,  2062,  2084]])
        }
    elif dataset_name == "cora":
        return {
            "degree_1": torch.tensor([[    2,   119,   130,   159,   193,   295,   325,   328,   336,   394,   515,   541,   561,   673,   736,   761,  1146,  1154,  1664,  1729],
                                      [  104,   199,   236,   281,   304,   515,   558,   663,   865,   957,  1171,  1641,  1668,  1749,  1908,  1958,  1975,  2107,  2204,  2438],
                                      [   55,   119,   178,   205,   458,   459,   722,   723,   761,  1079,  1139,  1349,  1615,  1755,  1784,  1888,  1912,  2088,  2270,  2310],
                                      [  166,   495,   553,   564,   623,   655,   663,   679,   895,   967,  1098,  1232,  1318,  1390,  1737,  2183,  2334,  2404,  2429,  2456],
                                      [   48,   201,   234,   315,   360,   389,   431,   625,   743,  1079,  1488,  1599,  1655,  1755,  1908,  1959,  1991,  2071,  2116,  2456]]),
            "degree_2": torch.tensor([[  311,   407,   508,   574,   674,   687,   961,  1008,  1057,  1180,  1656,  1659,  1680,  1748,  1952,  2002,  2094,  2167,  2223,  2241],
                                      [   73,    94,   120,   124,   202,   265,   300,   452,   456,   550,   614,   734,  1083,  1674,  1941,  2173,  2273,  2280,  2315,  2379],
                                      [  270,   301,   407,   538,   552,   602,   621,   644,   671,   746,  1097,  1810,  1829,  1869,  2048,  2133,  2190,  2291,  2397,  2418],
                                      [   18,    22,   128,   212,   265,   472,   614,   657,   685,   689,   740,  1152,  1212,  1320,  1404,  1733,  1811,  1833,  2168,  2277],
                                      [  190,   192,   252,   451,   548,   780,   888,  1070,  1083,  1130,  1198,  1306,  1352,  1358,  1466,  1570,  1638,  1833,  1904,  2272]]),
            "degree_3": torch.tensor([[   19,   127,   136,   156,   182,   224,   332,   383,   695,   720,  1010,  1250,  1283,  1468,  1560,  1861,  1930,  2084,  2288,  2316],
                                      [   23,   122,   483,   486,   547,   581,  1144,  1190,  1223,  1421,  1472,  1537,  1658,  1673,  1890,  1923,  2056,  2061,  2134,  2304],
                                      [  195,   343,   357,   505,   758,   853,   946,   991,  1009,  1259,  1328,  1332,  1415,  1502,  1560,  1630,  1632,  1740,  2274,  2345],
                                      [    3,   238,   297,   513,   851,  1030,  1195,  1251,  1529,  1607,  1630,  1831,  1863,  1917,  2040,  2159,  2219,  2389,  2457,  2464],
                                      [    3,    10,   648,   775,   998,  1003,  1014,  1107,  1455,  1489,  1502,  1658,  1862,  1901,  1949,  2017,  2050,  2295,  2344,  2435]]),
            "degree_5": torch.tensor([[   69,   189,   757,   811,   832,   943,  1047,  1176,  1407,  1569,  1657,  1754,  1895,  1899,  2075,  2229,  2372,  2384,  2386,  2473],
                                      [   30,   521,   615,   912,  1354,  1407,  1547,  1573,  1670,  1672,  1754,  1884,  1903,  1931,  1961,  2172,  2236,  2237,  2384,  2411],
                                      [  269,   284,   600,   757,   900,   943,  1013,  1036,  1121,  1147,  1184,  1511,  1735,  1754,  1944,  2172,  2333,  2372,  2428,  2461],
                                      [    0,   278,   381,   642,   781,   943,  1361,  1478,  1479,  1481,  1754,  1788,  1802,  1814,  1895,  2125,  2237,  2328,  2333,  2426],
                                      [   30,   382,  1047,  1134,  1263,  1298,  1496,  1535,  1701,  1788,  1814,  1870,  1972,  2026,  2172,  2257,  2258,  2376,  2385,  2445]]),
            "degree_8_to_10": torch.tensor([[  840,   856,   857,   916,  1007,  1172,  1189,  1193,  1304,  1317,  1443,  1608,  1838,  1976,  2013,  2149,  2181,  2184,  2390,  2394],
                                            [  638,   869,   876,  1064,  1172,  1193,  1304,  1388,  1430,  1460,  1536,  1710,  1840,  1877,  1976,  2012,  2013,  2170,  2265,  2313],
                                            [  879,   916,  1007,  1061,  1188,  1280,  1376,  1388,  1779,  1840,  1865,  1905,  2149,  2161,  2170,  2213,  2262,  2332,  2390,  2394],
                                            [  868,   938,  1021,  1061,  1172,  1193,  1375,  1388,  1460,  1523,  1532,  1536,  1838,  1840,  1976,  1992,  2051,  2149,  2185,  2275],
                                            [   38,   844,   868,   876,   938,  1042,  1193,  1257,  1375,  1376,  1443,  1476,  1536,  1687,  1710,  1840,  1905,  1992,  2161,  2322]]),
            "degree_15_to_25": torch.tensor([[   11,    13,   812,   838,   852,   989,  1053,  1209,  1238,  1255,  1271,  1347,  1491,  1558,  1622,  1639,  1746,  2044,  2082,  2381],
                                             [   11,    56,   838,   866,   989,  1005,  1179,  1209,  1271,  1491,  1497,  1558,  1576,  1622,  1639,  1746,  1820,  2082,  2346,  2366],
                                             [   11,    13,    56,   838,  1005,  1209,  1271,  1338,  1340,  1480,  1497,  1558,  1576,  1622,  1639,  1746,  1822,  2044,  2082,  2346],
                                             [   11,    13,   852,   866,   989,  1005,  1053,  1238,  1338,  1340,  1347,  1480,  1491,  1558,  1639,  1689,  1822,  2044,  2082,  2381],
                                             [   13,   812,   838,   852,   866,  1005,  1179,  1209,  1255,  1271,  1338,  1480,  1491,  1497,  1499,  1576,  1622,  1689,  1746,  2346]])
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    # @formatter:on
