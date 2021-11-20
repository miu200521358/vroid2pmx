# -*- coding: utf-8 -*-
#
import hashlib
import json

from mmd.PmxReader import PmxReader
from mmd.PmxData import PmxModel, Bone, RigidBody, Vertex, Material, Morph, DisplaySlot, RigidBody, Joint, Ik, IkLink, Bdef1, Bdef2, Bdef4, Sdef, Qdef, MaterialMorphData, UVMorphData, BoneMorphData, VertexMorphOffset, GroupMorphData # noqa
from module.MMath import MRect, MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from service.VroidExportService import BONE_PAIRS
from utils.MLogger import MLogger # noqa
from utils.MException import SizingException, MKilledException, MParseException     # noqa

logger = MLogger(__name__, level=1)


class VroidReader(PmxReader):
    def __init__(self, file_path):
        self.file_path = file_path
        self.offset = 0
        self.buffer = None

    def read_model_name(self):
        return ""

    def read_data(self):
        # Pmxモデル生成
        pmx = PmxModel()
        pmx.path = self.file_path

        try:
            with open(self.file_path, "rb") as f:
                self.buffer = f.read()

                signature = self.unpack(12, "12s")
                logger.test("signature: %s (%s)", signature, self.offset)

                # JSON文字列読み込み
                json_buf_size = self.unpack(8, "L")
                json_text = self.read_text(json_buf_size)

                pmx.json_data = json.loads(json_text)

                for material in pmx.json_data['materials']:
                    pmx.materials[material['name']] = Material(material['name'], material['name'], MVector3D(), -1, -1, MVector3D(), -1, -1, MVector3D(), 1, MVector4D(), -1, -1, -1, 1, 1)

                for bone_param in BONE_PAIRS.values():
                    pmx.bones[bone_param['name']] = Bone(bone_param['name'], bone_param['name'], MVector3D(), -1, -1, -1)

            # ハッシュを設定
            pmx.digest = self.hexdigest()
            logger.test("pmx: %s, hash: %s", pmx.name, pmx.digest)

            return pmx
        except MKilledException as ke:
            # 終了命令
            raise ke
        except SizingException as se:
            logger.error("Vroid読み込み処理が処理できないデータで終了しました。\n\n%s", se.message, decoration=MLogger.DECORATION_BOX)
            return se
        except Exception as e:
            import traceback
            logger.error("Vroid読み込み処理が意図せぬエラーで終了しました。\n\n%s", traceback.format_exc(), decoration=MLogger.DECORATION_BOX)
            raise e

    def read_text(self, format_size):
        bresult = self.unpack(format_size, "{0}s".format(format_size))
        return bresult.decode("UTF8")
