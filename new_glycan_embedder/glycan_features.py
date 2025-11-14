import warnings
from typing import List, Dict, Any

try:
    from glycowork.motif import tokenization
    GLYCOWORK_AVAILABLE = True
except ImportError:
    GLYCOWORK_AVAILABLE = False

UNIT2CATEGORY = {
    'ManHep': 'Hep', 'GalHep': 'Hep', 'DDGlcHep': 'Hep', 'DLGlcHep': 'Hep', 'IdoHep': 'Hep', 'LDManHep': 'Hep',
    'DDManHep': 'Hep', 'LyxHep': 'Hep', 'DDAltHep': 'Hep', 'Man': 'Hex', 'Glc': 'Hex', 'Galf': 'Hex', 'Hex': 'Hex',
    'Ido': 'Hex', 'Tal': 'Hex', 'Gul': 'Hex', 'All': 'Hex', 'Manf': 'Hex', 'Alt': 'Hex', 'Gal': 'Hex', 'Ins': 'Hex',
    'GulA': 'HexA', 'GalA': 'HexA', 'AltA': 'HexA', 'TalA': 'HexA', 'AllA': 'HexA', 'IdoA': 'HexA', 'GlcA': 'HexA',
    'HexA': 'HexA', 'ManA': 'HexA', 'ManN': 'HexN', 'AltN': 'HexN', 'GalN': 'HexN', 'HexN': 'HexN', 'AllN': 'HexN',
    'TalN': 'HexN', 'GulN': 'HexN', 'GlcN': 'HexN', 'GlcNAc': 'HexNAc', 'ManNAc': 'HexNAc', 'IdoNAc': 'HexNAc',
    'GlcfNAc': 'HexNAc', 'AltNAc': 'HexNAc', 'TalNAc': 'HexNAc', 'ManfNAc': 'HexNAc', 'GulNAc': 'HexNAc',
    'GalNAc': 'HexNAc', 'GalfNAc': 'HexNAc', 'AllNAc': 'HexNAc', 'HexNAc': 'HexNAc', 'Sor': 'Ket', 'Tag': 'Ket',
    'Fruf': 'Ket', 'Psi': 'Ket', 'Sedf': 'Ket', 'Fru': 'Ket', 'Xluf': 'Ket', 'Mur': 'Others', 'Erwiniose': 'Others',
    'MurNAc': 'Others', 'Pse': 'Others', 'Dha': 'Others', 'Fus': 'Others', 'Ko': 'Others', 'Pau': 'Others',
    'Aco': 'Others', 'IdoNGlcf': 'Others', 'dNon': 'Others', 'MurNGc': 'Others', 'ddNon': 'Others', 'Aci': 'Others',
    'Leg': 'Others', 'AcoNAc': 'Others', 'Api': 'Others', 'Apif': 'Others', 'Kdof': 'Others', 'Bac': 'Others',
    'Kdo': 'Others', 'Yer': 'Others', '4eLeg': 'Others', 'Ribf': 'Pen', 'Rib': 'Pen', 'Xyl': 'Pen', 'Ara': 'Pen',
    'Araf': 'Pen', 'Lyxf': 'Pen', 'Pen': 'Pen', 'Lyx': 'Pen', 'Xylf': 'Pen', 'AraN': 'PenN', 'Sia': 'Sia',
    'Neu5Ac': 'Sia', 'Neu5Gc': 'Sia', 'Neu': 'Sia', 'Kdn': 'Sia', 'Neu4Ac': 'Sia', 'Thre-ol': 'Tetol', 'Ery-ol': 'Tetol',
    '6dAltf': 'dHex', 'dHex': 'dHex', 'Qui': 'dHex', 'Rha': 'dHex', 'RhaN': 'dHex', 'Fuc': 'dHex', '6dAlt': 'dHex',
    'Fucf': 'dHex', '6dGul': 'dHex', '6dTal': 'dHex', 'QuiN': 'dHexN', 'FucN': 'dHexN', 'QuiNAc': 'dHexNAc',
    '6dTalNAc': 'dHexNAc', '6dAltNAc': 'dHexNAc', 'dHexNAc': 'dHexNAc', 'RhaNAc': 'dHexNAc', 'FucNAc': 'dHexNAc',
    'FucfNAc': 'dHexNAc', 'Par': 'ddHex', 'Asc': 'ddHex', 'Col': 'ddHex', 'Tyv': 'ddHex', 'Dig': 'ddHex', 'Oli': 'ddHex',
    'ddHex': 'ddHex', 'Abe': 'ddHex'
}


def get_core(token: str) -> str:
    if GLYCOWORK_AVAILABLE:
        return tokenization.get_core(token)
    else:
        return token


def onehot(x: str, vocab: List[str], allow_unknown: bool = False) -> List[int]:
    if x in vocab:
        index = vocab.index(x)
    else:
        index = -1

    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn(f"Unknown value `{x}`")
            feature[-1] = 1
        else:
            feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError(f"Unknown value `{x}`. Available vocabulary is `{vocab}`")
        feature[index] = 1

    return feature


class UnitFeatureExtractor:
    def __init__(self, unit_vocab: List[str]):
        self.unit_vocab = unit_vocab
        self.category_vocab = list(set(UNIT2CATEGORY.values()))

    def symbol(self, unit: str) -> List[int]:
        core_unit = get_core(unit)
        return onehot(core_unit, self.unit_vocab)

    def default(self, unit: str) -> List[int]:
        core_unit = get_core(unit)
        category = UNIT2CATEGORY.get(core_unit, "unknown_category")
        return self.symbol(unit) + onehot(category, self.category_vocab, allow_unknown=True)


class LinkFeatureExtractor:
    def __init__(self, link_vocab: List[str]):
        self.link_vocab = link_vocab

    def default(self, link: str) -> List[int]:
        core_link = get_core(link)
        return onehot(core_link, self.link_vocab)


class GlycanFeatureExtractor:
    def __init__(self):
        pass

    def default(self, iupac: str) -> List[float]:
        return [
            float(len(iupac)),
            float(iupac.count('(')),
            float(iupac.count('[')),
        ]


class FeatureRegistry:
    def __init__(self, unit_vocab: List[str], link_vocab: List[str]):
        self.unit_extractor = UnitFeatureExtractor(unit_vocab)
        self.link_extractor = LinkFeatureExtractor(link_vocab)
        self.glycan_extractor = GlycanFeatureExtractor()

        self.unit_features = {
            'symbol': self.unit_extractor.symbol,
            'default': self.unit_extractor.default
        }

        self.link_features = {
            'default': self.link_extractor.default
        }

        self.glycan_features = {
            'default': self.glycan_extractor.default
        }

    def get_unit_feature_func(self, name: str):
        if name not in self.unit_features:
            raise ValueError(f"Unknown unit feature: {name}. Available: {list(self.unit_features.keys())}")
        return self.unit_features[name]

    def get_link_feature_func(self, name: str):
        if name not in self.link_features:
            raise ValueError(f"Unknown link feature: {name}. Available: {list(self.link_features.keys())}")
        return self.link_features[name]

    def get_glycan_feature_func(self, name: str):
        if name not in self.glycan_features:
            raise ValueError(f"Unknown glycan feature: {name}. Available: {list(self.glycan_features.keys())}")
        return self.glycan_features[name]
