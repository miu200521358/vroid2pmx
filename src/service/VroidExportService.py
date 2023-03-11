# -*- coding: utf-8 -*-
#
import logging
from operator import index
import traceback
from PIL import Image, ImageChops
import struct
import os
import json
from pathlib import Path
import shutil
import numpy as np
import re
import math
import random
import string
import copy
import itertools

# import _pickle as cPickle

from module.MOptions import MExportOptions
from mmd.PmxData import (
    PmxModel,
    Vertex,
    Material,
    Bone,
    Morph,
    DisplaySlot,
    RigidBody,
    Joint,
    Bdef1,
    Bdef2,
    Bdef4,
    Sdef,
    RigidBodyParam,
    IkLink,
    Ik,
    BoneMorphData,
)
from mmd.PmxData import Bdef1, Bdef2, Bdef4, VertexMorphOffset, GroupMorphData, MaterialMorphData
from mmd.PmxWriter import PmxWriter
from mmd.VmdData import (
    VmdMotion,
    VmdBoneFrame,
    VmdCameraFrame,
    VmdInfoIk,
    VmdLightFrame,
    VmdMorphFrame,
    VmdShadowFrame,
    VmdShowIkFrame,
)
from module.MMath import MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4, MQuaternion
from utils import MServiceUtils, MFileUtils
from utils.MLogger import MLogger
from utils.MException import SizingException, MKilledException

logger = MLogger(__name__, level=1)

MIME_TYPE = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/ktx": "ktx",
    "image/ktx2": "ktx2",
    "image/webp": "webp",
    "image/vnd-ms.dds": "dds",
    "audio/wav": "wav",
}

# MMDにおける1cm＝0.125(ミクセル)、1m＝12.5
MIKU_METER = 12.5


class VroidExportService:
    def __init__(self, options: MExportOptions):
        self.options = options
        self.offset = 0
        self.buffer = None

    def execute(self):
        logging.basicConfig(level=self.options.logging_level, format="%(message)s [%(module_name)s]")

        try:
            service_data_txt = f"{logger.transtext('Vroid2Pmx処理実行')}\n------------------------\n{logger.transtext('exeバージョン')}: {self.options.version_name}\n"
            service_data_txt = (
                f"{service_data_txt}　{logger.transtext('元モデル')}: {os.path.basename(self.options.vrm_model.path)}\n"
            )

            logger.info(service_data_txt, translate=False, decoration=MLogger.DECORATION_BOX)

            model = self.vroid2pmx()
            if not model:
                return False

            # 最後に出力
            logger.info("PMX出力開始", decoration=MLogger.DECORATION_LINE)

            os.makedirs(os.path.dirname(self.options.output_path), exist_ok=True)
            PmxWriter().write(model, self.options.output_path)

            logger.info(
                "出力終了: %s", os.path.basename(self.options.output_path), decoration=MLogger.DECORATION_BOX, title="成功"
            )

            return True
        except MKilledException:
            return False
        except SizingException as se:
            logger.error("Vroid2Pmx処理が処理できないデータで終了しました。\n\n%s", se.message, decoration=MLogger.DECORATION_BOX)
        except Exception:
            logger.critical(
                "Vroid2Pmx処理が意図せぬエラーで終了しました。\n\n%s", traceback.format_exc(), decoration=MLogger.DECORATION_BOX
            )
        finally:
            logging.shutdown()

    def vroid2pmx(self):
        try:
            model, tex_dir_path, setting_dir_path = self.create_model()
            if not model:
                return False

            model, bone_name_dict = self.convert_bone(model)
            if not model:
                return False

            model = self.convert_mesh(model, bone_name_dict, tex_dir_path)
            if not model:
                return False

            model = self.reconvert_bone(model)
            if not model:
                return False

            model = self.convert_morph(model)
            if not model:
                return False

            model = self.transfer_stance(model)
            if not model:
                return False

            model = self.create_body_rigidbody(model)
            if not model:
                return False

            self.export_pmxtailor_setting(model, setting_dir_path)

            return model
        except MKilledException as ke:
            # 終了命令
            raise ke
        except SizingException as se:
            logger.error("Vroid2Pmx処理が処理できないデータで終了しました。\n\n%s", se.message, decoration=MLogger.DECORATION_BOX)
            return se
        except Exception as e:
            import traceback

            logger.critical(
                "Vroid2Pmx処理が意図せぬエラーで終了しました。\n\n%s", traceback.format_exc(), decoration=MLogger.DECORATION_BOX
            )
            raise e

    def export_pmxtailor_setting(self, model: PmxModel, setting_dir_path: str):
        if (
            "extensions" not in model.json_data
            or "VRM" not in model.json_data["extensions"]
            or "secondaryAnimation" not in model.json_data["extensions"]["VRM"]
            or "boneGroups" not in model.json_data["extensions"]["VRM"]["secondaryAnimation"]
            or "nodes" not in model.json_data
        ):
            return

        # 材質・ボーン・頂点INDEXの対応表を作成
        logger.info("-- PmxTailor用設定ファイル出力準備1")
        bone_materials = {}
        material_bones = {}
        for bone_idx, bone_vidxs in model.vertices.items():
            bone_name = model.bone_indexes.get(bone_idx, None)
            for material_name, vidxs in model.material_vertices.items():
                # 一定以上ウェイトが乗っている場合のみ対象とする
                if [
                    vidx
                    for vidx in list(set(vidxs) & set(bone_vidxs))
                    if bone_idx in model.vertex_dict[vidx].deform.get_idx_list(0.3)
                ]:
                    if bone_name not in bone_materials:
                        bone_materials[bone_name] = []
                    if material_name not in bone_materials[bone_name]:
                        bone_materials[bone_name].append(material_name)

                    if material_name not in material_bones:
                        material_bones[material_name] = []
                    if bone_idx not in material_bones[material_name]:
                        material_bones[material_name].append(bone_idx)

            logger.info("-- -- PmxTailor用設定ファイル出力準備1 (%s)", bone_name)

        HAIR_NAME = "髪"
        CAT_EAR_NAME = "CatEar2"
        RABBIT_EAR_NAME = "RabbitEar2"

        hair_bones = {}
        cat_ear_bones = {}
        rabbit_ear_bones = {}
        clothing_bones = {}

        for bname in model.bones.keys():
            if HAIR_NAME in bname:
                if bname[:4] not in hair_bones:
                    hair_bones[bname[:4]] = []
                hair_bones[bname[:4]].append(bname)

            if "装飾_" in bname:
                if bname[:9] not in clothing_bones:
                    clothing_bones[bname[:9]] = []
                clothing_bones[bname[:9]].append(bname)

            if CAT_EAR_NAME in bname:
                bkey = bname[bname.find(CAT_EAR_NAME) - 3 : bname.find(CAT_EAR_NAME) + len(CAT_EAR_NAME)]
                if bkey not in cat_ear_bones:
                    cat_ear_bones[bkey] = []
                if 2 > len(cat_ear_bones[bkey]):
                    cat_ear_bones[bkey].append(bname)

            if RABBIT_EAR_NAME in bname:
                bkey = bname[bname.find(RABBIT_EAR_NAME) - 3 : bname.find(RABBIT_EAR_NAME) + len(RABBIT_EAR_NAME)]
                if bkey not in rabbit_ear_bones:
                    rabbit_ear_bones[bkey] = []
                if 2 > len(rabbit_ear_bones[bkey]):
                    rabbit_ear_bones[bkey].append(bname)

        node_bone_names = {}
        for nidx, node in enumerate(model.json_data["nodes"]):
            for bone in model.bones.values():
                if bone.english_name == node["name"]:
                    node_bone_names[nidx] = bone.name
                    break

        logger.info("-- PmxTailor用設定ファイル出力準備2")

        bone_cnt = {"Bust": 1, "CatEar": 1, "RabbitEar": 1, "Sleeve": 1, "HoodString": 1, "Hood": 1}
        pmx_tailor_settings = {}
        for bone_group in model.json_data["extensions"]["VRM"]["secondaryAnimation"]["boneGroups"]:
            if bone_group["comment"] in ("Bust", "CatEar", "RabbitEar", "Sleeve", "HoodString", "Hood"):
                for bone_bidx in bone_group["bones"]:
                    bone_name = node_bone_names[bone_bidx]
                    bone = model.bones[bone_name]

                    if "Bust" == bone_group["comment"]:
                        target_names = ["CLOTH", "SKIN"]
                        abb_name = bone.name
                        parent_bone_name = "上半身3"
                        group = "1"

                        primitive_name = (
                            logger.transtext("胸(小)")
                            if bone.position.distanceToPoint(model.bones[model.bone_indexes[bone.tail_index]].position)
                            < 0.7
                            else logger.transtext("胸(大)")
                        )

                        target_bones = [[bone.name, f"{bone.name}先"]]
                    elif "CatEar" == bone_group["comment"]:
                        target_names = [CAT_EAR_NAME[:-1]]
                        abb_name = f"左猫耳" if "_L_" in bone.name else f"右猫耳"
                        parent_bone_name = "頭"
                        group = "1"
                        primitive_name = logger.transtext("髪(ショート)")
                        target_bones = [cat_ear_bones[("_L_CatEar2" if "_L_" in bone.name else "_R_CatEar2")]]
                    elif "RabbitEar" == bone_group["comment"]:
                        target_names = [RABBIT_EAR_NAME[:-1]]
                        abb_name = f"左兎耳" if "_L_" in bone.name else f"右兎耳"
                        parent_bone_name = "頭"
                        group = "1"
                        primitive_name = logger.transtext("髪(ショート)")
                        target_bones = [rabbit_ear_bones[("_L_RabbitEar2" if "_L_" in bone.name else "_R_RabbitEar2")]]
                    elif "Sleeve" == bone_group["comment"]:
                        target_names = ["CLOTH"]
                        if "LowerSleeve" in bone.name:
                            abb_name = f"左袖{bone_cnt['Sleeve']}" if "_L_" in bone.name else f"右袖{bone_cnt['Sleeve']}"
                            parent_bone_name = "左ひじ" if "_L_" in bone.name else "右ひじ"
                        elif "TipSleeve" in bone.name:
                            abb_name = f"左袖口{bone_cnt['Sleeve']}" if "_L_" in bone.name else f"右袖口{bone_cnt['Sleeve']}"
                            parent_bone_name = "左手首" if "_L_" in bone.name else "左手首"
                        bone_cnt["Sleeve"] += 1
                        primitive_name = logger.transtext("単一揺れ物")
                        group = "3"
                        target_bones = [[bone.name, model.bone_indexes[bone.tail_index]]]
                    elif "HoodString" == bone_group["comment"]:
                        target_names = ["CLOTH"]
                        abb_name = (
                            f"左紐{bone_cnt['HoodString']}" if "_L_" in bone.name else f"右紐{bone_cnt['HoodString']}"
                        )
                        bone_cnt["HoodString"] += 1
                        parent_bone_name = "上半身3"
                        group = "4"
                        primitive_name = logger.transtext("単一揺れ物")
                        target_bones = [
                            [model.bone_indexes[bone.parent_index], bone.name, model.bone_indexes[bone.tail_index]]
                        ]
                    elif "Hood" == bone_group["comment"]:
                        target_names = ["CLOTH"]
                        abb_name = f"フード{bone_cnt['Hood']}"
                        bone_cnt["Hood"] += 1
                        parent_bone_name = "首"
                        group = "4"
                        primitive_name = logger.transtext("単一揺れ物")
                        target_bones = [[bone.name, model.bone_indexes[bone.tail_index]]]

                    weighted_material_name = None
                    for target_name in target_names:
                        for material_name in bone_materials.get(bone.name, []):
                            if target_name in material_name:
                                weighted_material_name = model.materials[material_name].name
                                break
                        if weighted_material_name:
                            break

                    back_material_names = []
                    if f"{weighted_material_name}_エッジ" in model.materials:
                        back_material_names.append(f"{weighted_material_name}_エッジ")
                    if f"{weighted_material_name}_裏" in model.materials:
                        back_material_names.append(f"{weighted_material_name}_裏")

                    pmx_tailor_settings[bone.name] = {
                        "material_name": weighted_material_name,
                        "abb_name": abb_name,
                        "parent_bone_name": parent_bone_name,
                        "group": group,
                        "direction": logger.transtext("下"),
                        "primitive": primitive_name,
                        "exist_physics_clear": logger.transtext("再利用"),
                        "target_bones": target_bones,
                        "back_extend_material_names": back_material_names,
                        "rigidbody_root_thick": (0.3 if "紐" in abb_name else 0.5),
                        "rigidbody_end_thick": 0.5,
                    }

                    logger.info("-- -- PmxTailor用設定ファイル出力準備2 (%s)", abb_name)

        HAIR_AHOGE = logger.transtext("髪(アホ毛)")
        HAIR_SHORT = logger.transtext("髪(ショート)")
        HAIR_LONG = logger.transtext("髪(ロング)")
        ahoge_cnt = 1
        short_cnt = 1
        long_cnt = 1

        for bname, hbones in hair_bones.items():
            material_name = bone_materials.get(hbones[0], [""])[0]
            material_name = model.materials[material_name].name if material_name else None
            if len(hbones) > 1 and (model.bones[hbones[0]].position - model.bones[hbones[1]].position).y() < 0:
                if (HAIR_AHOGE, material_name) not in pmx_tailor_settings:
                    pmx_tailor_settings[(HAIR_AHOGE, material_name)] = {
                        "material_name": material_name,
                        "abb_name": f"髪H{ahoge_cnt}",
                        "parent_bone_name": "頭",
                        "group": "4",
                        "direction": logger.transtext("下"),
                        "primitive": HAIR_AHOGE,
                        "exist_physics_clear": logger.transtext("再利用"),
                        "target_bones": [hbones],
                        "back_extend_material_names": [],
                        "rigidbody_root_thick": 0.2,
                        "rigidbody_end_thick": 0.4,
                    }
                    logger.info("-- -- PmxTailor用設定ファイル出力準備2 (%s)", f"髪H{ahoge_cnt}")
                    ahoge_cnt += 1
                else:
                    pmx_tailor_settings[(HAIR_AHOGE, material_name)]["target_bones"].append(hbones)
            elif len(hbones) < 4:
                if (HAIR_SHORT, material_name) not in pmx_tailor_settings:
                    pmx_tailor_settings[(HAIR_SHORT, material_name)] = {
                        "material_name": material_name,
                        "abb_name": f"髪S{short_cnt}",
                        "parent_bone_name": "頭",
                        "group": "4",
                        "direction": logger.transtext("下"),
                        "primitive": HAIR_SHORT,
                        "exist_physics_clear": logger.transtext("再利用"),
                        "target_bones": [hbones],
                        "back_extend_material_names": [],
                        "rigidbody_root_thick": 0.3,
                        "rigidbody_end_thick": 1.2,
                    }
                    logger.info("-- -- PmxTailor用設定ファイル出力準備2 (%s)", f"髪S{short_cnt}")
                    short_cnt += 1
                else:
                    pmx_tailor_settings[(HAIR_SHORT, material_name)]["target_bones"].append(hbones)
            else:
                if (HAIR_LONG, material_name) not in pmx_tailor_settings:
                    pmx_tailor_settings[(HAIR_LONG, material_name)] = {
                        "material_name": material_name,
                        "abb_name": f"髪L{long_cnt}",
                        "parent_bone_name": "頭",
                        "group": "4",
                        "direction": logger.transtext("下"),
                        "primitive": HAIR_LONG,
                        "exist_physics_clear": logger.transtext("再利用"),
                        "target_bones": [hbones],
                        "back_extend_material_names": [],
                        "rigidbody_root_thick": 0.2,
                        "rigidbody_end_thick": 0.5,
                    }
                    logger.info("-- -- PmxTailor用設定ファイル出力準備2 (%s)", f"髪L{long_cnt}")
                    long_cnt += 1
                else:
                    pmx_tailor_settings[(HAIR_LONG, material_name)]["target_bones"].append(hbones)

        CLOTHING_COATSKIRT = logger.transtext("単一揺れ物")
        CLOTHING_SKIRT = logger.transtext("単一揺れ物")
        CLOTHING_COAT = logger.transtext("単一揺れ物")
        clothing_cnt = {"_CoatSkirt": 1, "_Skirt": 1, "_Coat": 1}

        for weighted_material_name, cbones in material_bones.items():
            if "CLOTH" not in weighted_material_name or not cbones:
                continue

            logger.debug("weighted_material_name: %s", weighted_material_name)

            for target_name, target_primitive, target_group, target_abb in (
                ("_CoatSkirt", CLOTHING_COATSKIRT, 7, "CS"),
                ("_Skirt", CLOTHING_SKIRT, 8, "SK"),
                ("_Coat", CLOTHING_COAT, 9, "CT"),
            ):
                if [
                    bidx
                    for bidx in cbones
                    if target_name in model.bone_indexes[bidx] and model.bone_indexes[bidx] in bone_materials
                ]:

                    material_name = model.materials[weighted_material_name].name
                    back_material_names = []
                    if f"{material_name}_エッジ" in model.materials:
                        back_material_names.append(f"{material_name}_エッジ")
                    if f"{material_name}_裏" in model.materials:
                        back_material_names.append(f"{material_name}_裏")

                    for direction, parent_bone_name in (("L", "左足"), ("R", "右足")):
                        target_bones = []

                        for nidx, vertical_name in enumerate(
                            [
                                f"{direction}{target_name}Back",
                                f"{direction}{target_name}Side",
                                f"{direction}{target_name}Front",
                            ]
                        ):
                            is_reset = True
                            for bidx in sorted(cbones):
                                bname = model.bone_indexes[bidx]
                                if vertical_name in bname:
                                    if is_reset:
                                        target_bones.append([])
                                        is_reset = False
                                    target_bones[-1].append(bname)

                            if target_bones and target_bones[-1]:
                                while model.bones[target_bones[-1][-1]].tail_index >= 0:
                                    # 末端ボーンまでを入れる
                                    target_bones[-1].append(
                                        model.bone_indexes[model.bones[target_bones[-1][-1]].tail_index]
                                    )

                        if target_name == "_Coat" and not target_bones:
                            # CoatはCoatSkirtと被るので、うまく取れなければスルー
                            continue

                        # target_bone_cnts = [len(lbones) for lbones in target_bones]
                        # # 段差がある場合、揃えておく（VRoid Studioのスカートはサイドだけ長いとかがある）
                        # for n, bone_cnt in enumerate(target_bone_cnts):
                        #     for _ in range(bone_cnt, np.max(target_bone_cnts)):
                        #         target_bones[n].insert(0, "")

                        pmx_tailor_settings[
                            (
                                target_name,
                                direction,
                                clothing_cnt[target_name],
                                target_primitive,
                                weighted_material_name,
                            )
                        ] = {
                            "material_name": material_name,
                            "abb_name": f"{target_abb}{direction}{clothing_cnt[target_name]}",
                            "parent_bone_name": parent_bone_name,
                            "group": str(target_group),
                            "direction": logger.transtext("下"),
                            "primitive": target_primitive,
                            "exist_physics_clear": logger.transtext("再利用"),
                            "target_bones": target_bones,
                            "back_extend_material_names": back_material_names,
                            "rigidbody_root_thick": 0.35,
                            "rigidbody_end_thick": 0.5,
                        }

                        logger.info(
                            "-- -- PmxTailor用設定ファイル出力準備2 (%s)", ", ".join((target_name, direction, material_name))
                        )

                    clothing_cnt[target_name] += 1

        for setting_name, pmx_tailor_setting in pmx_tailor_settings.items():
            if pmx_tailor_setting["material_name"] and pmx_tailor_setting["target_bones"]:
                with open(
                    os.path.join(setting_dir_path, f"{pmx_tailor_setting['abb_name']}.json"), "w", encoding="utf-8"
                ) as jf:
                    json.dump(pmx_tailor_setting, jf, ensure_ascii=False, indent=4, separators=(",", ": "))
            else:
                logger.warning(
                    "VRoid Studioで設定された物理をPmxTailor用設定に変換できませんでした。 定義名: %s, 材質名: %s, ボーン名: %s",
                    setting_name,
                    pmx_tailor_setting["material_name"],
                    pmx_tailor_setting["target_bones"],
                )

        logger.info("-- PmxTailor用設定ファイル出力終了")

    def create_body_rigidbody(self, model: PmxModel):
        skin_vidxs = []
        cloth_vidxs = []
        for material_name, vidxs in model.material_vertices.items():
            if "SKIN" in model.materials[material_name].english_name:
                skin_vidxs.extend(vidxs)
            elif "CLOTH" in model.materials[material_name].english_name:
                cloth_vidxs.extend(vidxs)

        bone_vertices = {}
        bone_weights = {}
        for bidx, vidxs in model.vertices.items():
            bone = model.bones[model.bone_indexes[bidx]]
            target_bone_weights = {}

            bone_strong_vidxs = [
                vidx for vidx in vidxs if bone.index in model.vertex_dict[vidx].deform.get_idx_list(0.4)
            ]
            target_bone_vidxs = list(set(skin_vidxs) & set(bone_strong_vidxs))

            if 20 > len(target_bone_vidxs):
                # 強参照頂点が少ない場合、弱参照頂点を確認する
                bone_weak_vidxs = [
                    vidx for vidx in vidxs if bone.index in model.vertex_dict[vidx].deform.get_idx_list(0.2)
                ]
                target_bone_vidxs = list(set(skin_vidxs) & set(bone_weak_vidxs))

            if 20 > len(target_bone_vidxs) or "足先EX" in bone.name:
                # 弱参照肌頂点が少ない場合、衣装強参照頂点を確認する
                # 足先は靴が必ず入るので衣装も含む
                target_bone_vidxs = list((set(skin_vidxs) | set(cloth_vidxs)) & set(bone_strong_vidxs))

            if 20 > len(target_bone_vidxs):
                # 衣装強参照頂点が少ない場合、衣装弱参照頂点を確認する
                target_bone_vidxs = list((set(skin_vidxs) | set(cloth_vidxs)) & set(bone_weak_vidxs))

            if 20 > len(target_bone_vidxs):
                continue

            for vidx in target_bone_vidxs:
                target_bone_weights[vidx] = model.vertex_dict[vidx].deform.get_weight(bone.index)

            bones = []
            if "捩" in bone.name:
                # 捩りは親に入れる
                bones.append(model.bones[model.bone_indexes[bone.parent_index]])
            elif "指" in bone.name:
                bones.append(model.bones[f"{bone.name[0]}手首"])
            elif "胸先" in bone.name:
                bones.append(model.bones[f"{bone.name[0]}胸"])
            elif "胸" in bone.name:
                bones.append(bone)
                # 胸は上半身2にも割り振る
                bones.append(model.bones["上半身2"])
            elif "足先EX" in bone.name:
                bones.append(model.bones[f"{bone.name[0]}足首"])
            elif bone.getExternalRotationFlag() and bone.effect_factor == 1:
                # 回転付与の場合、付与親に入れる(足D系)
                bones.append(model.bones[model.bone_indexes[bone.effect_index]])
            else:
                bones.append(bone)

            # 導入対象に入れる
            for bone in bones:
                if bone.name not in bone_vertices:
                    bone_vertices[bone.name] = []
                    bone_weights[bone.name] = {}
                bone_vertices[bone.name].extend(target_bone_vidxs)
                for vidx, weight in target_bone_weights.items():
                    if vidx not in bone_weights[bone.name]:
                        bone_weights[bone.name][vidx] = 0
                    bone_weights[bone.name][vidx] += weight

        logger.info("-- 身体剛体準備終了")

        for rigidbody_name, rigidbody_param in RIGIDBODY_PAIRS.items():
            no_collision_group = 0
            for nc in range(16):
                if nc not in rigidbody_param["no_collision_group"]:
                    no_collision_group |= 1 << nc

            bone = model.bones[rigidbody_param["bone"]]

            # ボーンの向き先に沿う
            if "手首" in bone.name:
                # 手首は中指3を方向とする
                tail_position = model.bones[f"{bone.name[0]}中指先"].position
            else:
                if bone.tail_index > 0:
                    tail_bone = [b for b in model.bones.values() if bone.tail_index == b.index][0]
                    tail_position = tail_bone.position
                else:
                    tail_position = bone.tail_position + bone.position

            if rigidbody_param["direction"] == "horizonal":
                # ボーン進行方向(x)
                x_direction_pos = MVector3D(1, 0, 0)
                # ボーン進行方向に対しての横軸(y)
                y_direction_pos = MVector3D(0, 1, 0)
            else:
                # ボーン進行方向(x)
                x_direction_pos = (tail_position - bone.position).normalized()
                # ボーン進行方向に対しての横軸(y)
                y_direction_pos = MVector3D(1, 0, 0)
            # ボーン進行方向に対しての縦軸(z)
            z_direction_pos = MVector3D.crossProduct(x_direction_pos, y_direction_pos)
            bone_shape_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)

            mat = MMatrix4x4()
            mat.setToIdentity()
            mat.translate(bone.position)
            mat.rotate(bone_shape_qq)

            vposes = []
            vweights = []
            if "尻" in rigidbody_name:
                for vidx in bone_vertices.get("下半身", []):
                    if ("右" in rigidbody_name and model.vertex_dict[vidx].position.x() <= 0) or (
                        "左" in rigidbody_name and model.vertex_dict[vidx].position.x() >= 0
                    ):
                        vposes.append(model.vertex_dict[vidx].position.data())
                        vweights.append(bone_weights["下半身"][vidx])
            else:
                for vidx in bone_vertices.get(bone.name, []):
                    vposes.append(model.vertex_dict[vidx].position.data())
                    vweights.append(bone_weights[bone.name][vidx])

            if not vposes:
                continue

            if rigidbody_param["range"] in ["upper", "lower"]:
                # 重心
                gravity_pos = MVector3D(np.average(vposes, axis=0, weights=vweights))

                mat = MMatrix4x4()
                mat.setToIdentity()
                mat.translate(gravity_pos)
                mat.rotate(bone_shape_qq)

                # 上下に分ける系はローカル位置で分ける
                local_vposes = np.array([(mat.inverted() * MVector3D(vpos)).data() for vpos in vposes])

                # 中央値
                mean_y = np.mean(local_vposes, axis=0)[1]

                target_vposes = []
                target_vweights = []
                for vpos, vweight in zip(local_vposes, vweights):
                    if (vpos[1] >= mean_y and rigidbody_param["range"] == "upper") or (
                        vpos[1] <= mean_y and rigidbody_param["range"] == "lower"
                    ):
                        target_vposes.append((mat * MVector3D(vpos)).data())
                        target_vweights.append(vweight)
            else:
                target_vposes = vposes
                target_vweights = vweights

            # 重心
            shape_position = MVector3D(np.average(target_vposes, axis=0, weights=target_vweights))

            mat = MMatrix4x4()
            mat.setToIdentity()
            mat.translate(shape_position)
            mat.rotate(bone_shape_qq)

            target_local_vposes = np.array([(mat.inverted() * MVector3D(vpos)).data() for vpos in target_vposes])

            local_vpos_diff = np.max(target_local_vposes, axis=0) - np.min(target_local_vposes, axis=0)

            if rigidbody_param["shape"] == 0:
                x_size = y_size = np.mean(local_vpos_diff) / 2
            else:
                x_size = np.mean(local_vpos_diff[0::2]) / 2
                y_size = local_vpos_diff[1] - x_size * 0.7

            if rigidbody_name in ["上半身2", "上半身3"]:
                # ちょっと後ろにずらす
                shape_position.setZ(shape_position.z() + (x_size * 0.5))

            shape_size = MVector3D(x_size, y_size, x_size) * rigidbody_param.get("ratio", MVector3D(1, 1, 1))

            if rigidbody_param["shape"] == 0:
                # 球剛体はバウンティングボックスの中心
                shape_position = MVector3D(
                    np.mean([np.max(target_vposes, axis=0), np.min(target_vposes, axis=0)], axis=0)
                )
                if "尻" in rigidbody_name:
                    shape_position = MVector3D(
                        np.average(
                            [model.bones["下半身"].position.data(), model.bones[f"{rigidbody_name[0]}足"].position.data()],
                            axis=0,
                            weights=[0.3, 0.7],
                        )
                    )
                elif "胸" in rigidbody_name:
                    shape_position = MVector3D(
                        np.average(
                            [shape_position.data(), model.bones[f"{rigidbody_name[0]}胸"].position.data()],
                            axis=0,
                            weights=[0.3, 0.7],
                        )
                    )
                elif "後頭部" in rigidbody_name:
                    shape_position = MVector3D(np.average(target_vposes, axis=0, weights=target_vweights))
                    shape_position.setZ(shape_position.z() + (x_size * 0.3))
                elif "頭" in rigidbody_name:
                    shape_position = MVector3D(np.average(target_vposes, axis=0, weights=target_vweights))
                    shape_position.setY(shape_position.y() + (x_size * 0.3))
                    shape_position.setZ(shape_position.z() - (x_size * 0.1))

            if "足首" in rigidbody_name or "首" == rigidbody_name or "太もも" in rigidbody_name:
                mat = MMatrix4x4()
                mat.setToIdentity()
                mat.translate(shape_position)
                mat.rotate(bone_shape_qq)

                if "足首" in rigidbody_name:
                    shape_position = mat * MVector3D(0, -y_size * 0.15, x_size * 0.2)
                elif "首" == rigidbody_name:
                    shape_position = mat * MVector3D(0, 0, x_size * 0.3)
                elif "太もも" in rigidbody_name:
                    shape_position = mat * MVector3D(x_size * 0.1 * np.sign(bone.position.x()), 0, -x_size * 0.1)

            if rigidbody_param["direction"] == "horizonal":
                # ボーン進行方向(x)
                x_direction_pos = MVector3D(1, 0, 0)
                # ボーン進行方向に対しての横軸(y)
                y_direction_pos = MVector3D(0, 1, 0)
            elif rigidbody_param["direction"] == "vertical":
                # ボーン進行方向(x)
                x_direction_pos = (tail_position - bone.position).normalized()
                # ボーン進行方向に対しての横軸(y)
                y_direction_pos = MVector3D(1, 0, 0)
            else:
                # ボーン進行方向(x)
                x_direction_pos = (bone.position - tail_position).normalized()
                # ボーン進行方向に対しての横軸(y)
                y_direction_pos = MVector3D(-1, 0, 0)
            # ボーン進行方向に対しての縦軸(z)
            z_direction_pos = MVector3D.crossProduct(x_direction_pos, y_direction_pos)
            shape_qq = MQuaternion.fromDirection(z_direction_pos, x_direction_pos)
            shape_euler = shape_qq.toEulerAngles()
            shape_rotation_radians = MVector3D(
                math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z())
            )

            rigidbody = RigidBody(
                rigidbody_name,
                rigidbody_param["english"],
                bone.index,
                rigidbody_param["group"],
                no_collision_group,
                rigidbody_param["shape"],
                shape_size,
                shape_position,
                shape_rotation_radians,
                10,
                0.5,
                0.5,
                0,
                0,
                0,
            )
            rigidbody.index = len(model.rigidbodies)
            model.rigidbodies[rigidbody.name] = rigidbody

            logger.info("-- -- 身体剛体[%s]", rigidbody_name)

        logger.info("-- 身体剛体設定終了")

        return model

    def transfer_stance(self, model: PmxModel):
        # 各頂点
        all_vertex_relative_poses = {}
        for vertex in model.vertex_dict.values():
            if type(vertex.deform) is Bdef1:
                all_vertex_relative_poses[vertex.index] = [
                    vertex.position - model.bones[model.bone_indexes[vertex.deform.index0]].position
                ]
            elif type(vertex.deform) is Bdef2:
                all_vertex_relative_poses[vertex.index] = [
                    vertex.position - model.bones[model.bone_indexes[vertex.deform.index0]].position,
                    vertex.position - model.bones[model.bone_indexes[vertex.deform.index1]].position,
                ]
            elif type(vertex.deform) is Bdef4:
                all_vertex_relative_poses[vertex.index] = [
                    vertex.position - model.bones[model.bone_indexes[vertex.deform.index0]].position,
                    vertex.position - model.bones[model.bone_indexes[vertex.deform.index1]].position,
                    vertex.position - model.bones[model.bone_indexes[vertex.deform.index2]].position,
                    vertex.position - model.bones[model.bone_indexes[vertex.deform.index3]].position,
                ]

        trans_bone_vecs = {}
        trans_bone_mats = {}
        trans_vertex_vecs = {}
        trans_normal_vecs = {}

        trans_bone_vecs["全ての親"] = MVector3D()
        trans_bone_mats["全ての親"] = MMatrix4x4()
        trans_bone_mats["全ての親"].setToIdentity()

        bone_names = ["頭"]

        for direction in ["右", "左"]:
            bone_names.extend(
                [
                    f"{direction}親指先",
                    f"{direction}人指先",
                    f"{direction}中指先",
                    f"{direction}薬指先",
                    f"{direction}小指先",
                    f"{direction}胸先",
                    f"{direction}腕捩1",
                    f"{direction}腕捩2",
                    f"{direction}腕捩3",
                    f"{direction}手捩1",
                    f"{direction}手捩2",
                    f"{direction}手捩3",
                ]
            )

        # 装飾は人体の後
        for bname in model.bones.keys():
            if "装飾_" in bname:
                bone_names.append(bname)

        for end_bone_name in bone_names:
            bone_links = model.create_link_2_top_one(end_bone_name, is_defined=False).to_links("上半身")
            if len(bone_links.all().keys()) == 0:
                continue

            trans_vs = MServiceUtils.calc_relative_position(model, bone_links, VmdMotion(), 0)

            if "右" in ",".join(list(bone_links.all().keys())):
                arm_astance_qq = MQuaternion.fromEulerAngles(0, 0, 35)
                arm_bone_name = "右腕"
                thumb0_stance_qq = MQuaternion.fromEulerAngles(0, 8, 0)
                thumb0_bone_name = "右親指０"
                thumb1_stance_qq = MQuaternion.fromEulerAngles(0, 24, 0)
                thumb1_bone_name = "右親指１"
            elif "左" in ",".join(list(bone_links.all().keys())):
                arm_astance_qq = MQuaternion.fromEulerAngles(0, 0, -35)
                arm_bone_name = "左腕"
                thumb0_stance_qq = MQuaternion.fromEulerAngles(0, -8, 0)
                thumb0_bone_name = "左親指０"
                thumb1_stance_qq = MQuaternion.fromEulerAngles(0, -24, 0)
                thumb1_bone_name = "左親指１"
            else:
                arm_astance_qq = MQuaternion.fromEulerAngles(0, 0, 0)
                arm_bone_name = ""
                thumb0_bone_name = ""
                thumb1_bone_name = ""

            mat = MMatrix4x4()
            mat.setToIdentity()
            for vi, (bone_name, trans_v) in enumerate(zip(bone_links.all().keys(), trans_vs)):
                mat.translate(trans_v)
                if bone_name == arm_bone_name:
                    # 腕回転させる
                    mat.rotate(arm_astance_qq)
                elif bone_name == thumb0_bone_name:
                    # 親指0回転させる
                    mat.rotate(thumb0_stance_qq)
                elif bone_name == thumb1_bone_name:
                    # 親指1回転させる
                    mat.rotate(thumb1_stance_qq)

                if bone_name not in trans_bone_vecs:
                    trans_bone_vecs[bone_name] = mat * MVector3D()
                    trans_bone_mats[bone_name] = mat.copy()

        for bone_name, bone_vec in trans_bone_vecs.items():
            model.bones[bone_name].position = bone_vec

        local_y_vector = MVector3D(0, -1, 0)
        # local_z_vector = MVector3D(0, 0, -1)
        for bone_name in trans_bone_mats.keys():
            bone = model.bones[bone_name]
            direction = bone_name[0]
            arm_bone_name = f"{direction}腕"
            elbow_bone_name = f"{direction}ひじ"
            wrist_bone_name = f"{direction}手首"
            finger_bone_name = f"{direction}中指１"

            # ローカル軸
            if bone.name in ["右肩", "左肩"] and arm_bone_name in model.bones:
                bone.local_x_vector = (
                    model.bones[arm_bone_name].position - model.bones[bone.name].position
                ).normalized()
                bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()
            if bone.name in ["右腕", "左腕"] and elbow_bone_name in model.bones:
                bone.local_x_vector = (
                    model.bones[elbow_bone_name].position - model.bones[bone.name].position
                ).normalized()
                bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()
            if bone.name in ["右ひじ", "左ひじ"] and wrist_bone_name in model.bones:
                # ローカルYで曲げる
                bone.local_x_vector = (
                    model.bones[wrist_bone_name].position - model.bones[bone.name].position
                ).normalized()
                bone.local_z_vector = MVector3D.crossProduct(local_y_vector, bone.local_x_vector).normalized()
            if bone.name in ["右手首", "左手首"] and finger_bone_name in model.bones:
                bone.local_x_vector = (
                    model.bones[finger_bone_name].position - model.bones[bone.name].position
                ).normalized()
                bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()
            # 捩り
            if bone.name in ["右腕捩", "左腕捩"] and arm_bone_name in model.bones and elbow_bone_name in model.bones:
                bone.fixed_axis = (
                    model.bones[elbow_bone_name].position - model.bones[arm_bone_name].position
                ).normalized()
                bone.local_x_vector = (
                    model.bones[elbow_bone_name].position - model.bones[arm_bone_name].position
                ).normalized()
                bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()
            if bone.name in ["右手捩", "左手捩"] and elbow_bone_name in model.bones and wrist_bone_name in model.bones:
                bone.fixed_axis = (
                    model.bones[wrist_bone_name].position - model.bones[elbow_bone_name].position
                ).normalized()
                bone.local_x_vector = (
                    model.bones[wrist_bone_name].position - model.bones[elbow_bone_name].position
                ).normalized()
                bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()
            # 指
            if (
                bone.english_name in BONE_PAIRS
                and BONE_PAIRS[bone.english_name]["display"]
                and "指" in BONE_PAIRS[bone.english_name]["display"]
            ):
                bone.local_x_vector = (
                    model.bones[model.bone_indexes[bone.tail_index]].position
                    - model.bones[model.bone_indexes[bone.parent_index]].position
                ).normalized()
                bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()

        for vertex_idx, vertex_relative_poses in all_vertex_relative_poses.items():
            if vertex_idx not in trans_vertex_vecs:
                vertex = model.vertex_dict[vertex_idx]
                if type(vertex.deform) is Bdef1 and model.bone_indexes[vertex.deform.index0] in trans_bone_mats:
                    trans_vertex_vecs[vertex.index] = (
                        trans_bone_mats[model.bone_indexes[vertex.deform.index0]] * vertex_relative_poses[0]
                    )
                    trans_normal_vecs[vertex.index] = self.calc_normal(
                        trans_bone_mats[model.bone_indexes[vertex.deform.index0]], vertex.normal
                    )
                elif type(vertex.deform) is Bdef2 and (
                    model.bone_indexes[vertex.deform.index0] in trans_bone_mats
                    and model.bone_indexes[vertex.deform.index1] in trans_bone_mats
                ):
                    v0_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index0]] * vertex_relative_poses[0]
                    v1_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index1]] * vertex_relative_poses[1]
                    trans_vertex_vecs[vertex.index] = (v0_vec * vertex.deform.weight0) + (
                        v1_vec * (1 - vertex.deform.weight0)
                    )

                    v0_normal = self.calc_normal(
                        trans_bone_mats[model.bone_indexes[vertex.deform.index0]], vertex.normal
                    )
                    v1_normal = self.calc_normal(
                        trans_bone_mats[model.bone_indexes[vertex.deform.index1]], vertex.normal
                    )
                    trans_normal_vecs[vertex.index] = (v0_normal * vertex.deform.weight0) + (
                        v1_normal * (1 - vertex.deform.weight0)
                    )
                elif type(vertex.deform) is Bdef4 and (
                    model.bone_indexes[vertex.deform.index0] in trans_bone_mats
                    and model.bone_indexes[vertex.deform.index1] in trans_bone_mats
                    and model.bone_indexes[vertex.deform.index2] in trans_bone_mats
                    and model.bone_indexes[vertex.deform.index3] in trans_bone_mats
                ):
                    v0_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index0]] * vertex_relative_poses[0]
                    v1_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index1]] * vertex_relative_poses[1]
                    v2_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index2]] * vertex_relative_poses[2]
                    v3_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index3]] * vertex_relative_poses[3]
                    trans_vertex_vecs[vertex.index] = (
                        (v0_vec * vertex.deform.weight0)
                        + (v1_vec * vertex.deform.weight1)
                        + (v2_vec * vertex.deform.weight2)
                        + (v3_vec * vertex.deform.weight3)
                    )

                    v0_normal = self.calc_normal(
                        trans_bone_mats[model.bone_indexes[vertex.deform.index0]], vertex.normal
                    )
                    v1_normal = self.calc_normal(
                        trans_bone_mats[model.bone_indexes[vertex.deform.index1]], vertex.normal
                    )
                    v2_normal = self.calc_normal(
                        trans_bone_mats[model.bone_indexes[vertex.deform.index2]], vertex.normal
                    )
                    v3_normal = self.calc_normal(
                        trans_bone_mats[model.bone_indexes[vertex.deform.index3]], vertex.normal
                    )
                    trans_normal_vecs[vertex.index] = (
                        (v0_normal * vertex.deform.weight0)
                        + (v1_normal * vertex.deform.weight1)
                        + (v2_normal * vertex.deform.weight2)
                        + (v3_normal * vertex.deform.weight3)
                    )

        for (vertex_idx, vertex_vec), (_, vertex_normal) in zip(trans_vertex_vecs.items(), trans_normal_vecs.items()):
            model.vertex_dict[vertex_idx].position = vertex_vec
            model.vertex_dict[vertex_idx].normal = vertex_normal.normalized()

        logger.info("-- Aスタンス・親指調整終了")

        return model

    def calc_normal(self, bone_mat: MMatrix4x4, normal: MVector3D):
        # ボーン行列の3x3行列
        bone_invert_mat = bone_mat.data()[:3, :3]

        return MVector3D(np.sum(normal.data() * bone_invert_mat, axis=1)).normalized()

    def convert_morph(self, model: PmxModel):
        # グループモーフ定義
        if (
            "extensions" not in model.json_data
            or "VRM" not in model.json_data["extensions"]
            or "blendShapeMaster" not in model.json_data["extensions"]["VRM"]
            or "blendShapeGroups" not in model.json_data["extensions"]["VRM"]["blendShapeMaster"]
        ):
            return model

        # 一旦置き換えて、既存はクリア
        vertex_morphs = copy.deepcopy(model.org_morphs)
        target_morphs = copy.deepcopy(model.org_morphs)
        model.org_morphs = {}

        logger.info("-- -- モーフ調整準備")

        face_close_dict = {}
        for base_offset in target_morphs["Fcl_EYE_Close"].offsets:
            face_close_dict[base_offset.vertex_index] = base_offset.position_offset.copy().data()

        face_material_index_vertices = []
        face_left_close_index_vertices = []
        face_right_close_index_vertices = []
        for mat_name, mat_idxs in model.material_indices.items():
            if "_Face_" in mat_name:
                for index_idx in mat_idxs:
                    face_material_index_vertices.append(
                        [
                            model.vertex_dict[model.indices[index_idx][0]].position.data(),
                            model.vertex_dict[model.indices[index_idx][1]].position.data(),
                            model.vertex_dict[model.indices[index_idx][2]].position.data(),
                        ]
                    )

                    close_poses = [
                        model.vertex_dict[model.indices[index_idx][0]].position.data()
                        + face_close_dict.get(model.indices[index_idx][0], np.zeros(3)),
                        model.vertex_dict[model.indices[index_idx][1]].position.data()
                        + face_close_dict.get(model.indices[index_idx][1], np.zeros(3)),
                        model.vertex_dict[model.indices[index_idx][2]].position.data()
                        + face_close_dict.get(model.indices[index_idx][2], np.zeros(3)),
                    ]

                    if np.mean(close_poses, axis=0)[0] < 0:
                        face_right_close_index_vertices.append(close_poses)
                    else:
                        face_left_close_index_vertices.append(close_poses)
                break

        face_material_index_vertices = np.array(face_material_index_vertices)
        face_left_close_index_vertices = np.array(face_left_close_index_vertices)
        face_right_close_index_vertices = np.array(face_right_close_index_vertices)

        # 定義済みグループモーフ
        for sidx, shape in enumerate(model.json_data["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]):
            if len(shape["binds"]) == 0:
                continue

            if sidx > 0 and sidx % 10 == 0:
                logger.info("-- -- モーフ調整: %s個目", sidx)

            morph_name = shape["name"]
            morph_panel = 4
            if shape["name"] in MORPH_PAIRS:
                morph_name = MORPH_PAIRS[shape["name"]]["name"]
                morph_panel = MORPH_PAIRS[shape["name"]]["panel"]
            morph = Morph(morph_name, shape["name"], morph_panel, 0)
            morph.index = len(target_morphs)

            if shape["name"] in MORPH_PAIRS and "binds" in MORPH_PAIRS[shape["name"]]:
                for bind in MORPH_PAIRS[shape["name"]]["binds"]:
                    morph.offsets.append(GroupMorphData(target_morphs[bind].index, 1))
            else:
                for bind in shape["binds"]:
                    morph.offsets.append(GroupMorphData(bind["index"], bind["weight"] / 100))
            target_morphs[morph_name] = morph

        logger.info("-- -- モーフ調整: %s個目", sidx)

        # 自前グループモーフ
        for sidx, (morph_name, morph_pair) in enumerate(MORPH_PAIRS.items()):
            if sidx > 0 and sidx % 20 == 0:
                logger.info("-- -- 拡張モーフ調整: %s個目", sidx)

            if "binds" in morph_pair:
                # 統合グループモーフ（ある場合だけ）
                morph = Morph(morph_pair["name"], morph_name, morph_pair["panel"], 0)
                morph.index = len(target_morphs)
                ratios = (
                    morph_pair["ratios"] if "ratios" in morph_pair else [1 for _ in range(len(morph_pair["binds"]))]
                )
                for bind_name, bind_ratio in zip(morph_pair["binds"], ratios):
                    if bind_name in target_morphs:
                        bind_morph = target_morphs[bind_name]
                        morph.offsets.append(GroupMorphData(bind_morph.index, bind_ratio))
                if len(morph.offsets) > 0:
                    target_morphs[morph_name] = morph
            elif "split" in morph_pair:
                if morph_pair["split"] in target_morphs:
                    # 元のモーフを左右に分割する
                    org_morph = target_morphs[morph_pair["split"]]
                    target_offset = []
                    if org_morph.morph_type == 1:
                        if re.search(r"raiseEyelid_", morph_name):
                            # 目の上下で分けるタイプ
                            vposes = []
                            for offset in org_morph.offsets:
                                if offset.position_offset == MVector3D():
                                    continue
                                vertex = model.vertex_dict[offset.vertex_index]
                                vposes.append(vertex.position.data())
                            # モーフの中央
                            min_vertex = np.min(vposes, axis=0)
                            max_vertex = np.max(vposes, axis=0)
                            mean_vertex = np.mean(vposes, axis=0)
                            min_limit_y = np.mean([min_vertex[1], mean_vertex[1]])
                            max_limit_y = np.mean([max_vertex[1], mean_vertex[1]])
                            for offset in org_morph.offsets:
                                if offset.position_offset == MVector3D():
                                    continue
                                vertex = model.vertex_dict[offset.vertex_index]
                                if vertex.position.y() <= min_limit_y:
                                    ratio = (
                                        1
                                        if vertex.position.y() < max_limit_y
                                        else calc_ratio(vertex.position.y(), min_vertex[1], max_limit_y, 0, 1)
                                    )
                                    target_offset.append(
                                        VertexMorphOffset(offset.vertex_index, offset.position_offset * ratio)
                                    )
                        else:
                            for offset in org_morph.offsets:
                                if offset.position_offset == MVector3D():
                                    continue
                                vertex = model.vertex_dict[offset.vertex_index]
                                if ("_R" == morph_name[-2:] and vertex.position.x() < 0) or (
                                    "_L" == morph_name[-2:] and vertex.position.x() > 0
                                ):
                                    if morph_pair["panel"] == MORPH_LIP:
                                        # リップは中央にいくに従ってオフセットを弱める
                                        ratio = (
                                            1
                                            if abs(vertex.position.x()) >= 0.2
                                            else calc_ratio(abs(vertex.position.x()), 0, 0.2, 0, 1)
                                        )
                                        target_offset.append(
                                            VertexMorphOffset(offset.vertex_index, offset.position_offset * ratio)
                                        )
                                    else:
                                        target_offset.append(
                                            VertexMorphOffset(offset.vertex_index, offset.position_offset.copy())
                                        )
                    if target_offset:
                        morph = Morph(morph_pair["name"], morph_name, morph_pair["panel"], 1)
                        morph.index = len(target_morphs)
                        morph.offsets = target_offset

                        target_morphs[morph_name] = morph
            elif "creates" in morph_pair:
                # 生成タイプ
                target_material_vertices = []
                hide_material_vertices = []
                face_material_vertices = None
                for mat_name, mat_vert in model.material_vertices.items():
                    for create_mat_name in morph_pair["creates"]:
                        if create_mat_name in mat_name:
                            target_material_vertices.extend(mat_vert)
                    for hide_mat_name in morph_pair.get("hides", []):
                        if hide_mat_name in mat_name:
                            hide_material_vertices.extend(mat_vert)
                    if "_Face_" in mat_name:
                        face_material_vertices = mat_vert

                target_material_vertices = list(set(target_material_vertices))
                face_material_vertices = list(set(face_material_vertices))

                if target_material_vertices and face_material_vertices:
                    target_offset = []

                    # 処理対象の位置データ
                    target_poses = []
                    for vidx in target_material_vertices:
                        vertex = model.vertex_dict[vidx]
                        target_poses.append(vertex.position.data())

                    # 処理対象頂点を左右に分ける
                    left_target_poses = np.array(target_poses)[np.where(np.array(target_poses)[:, 0] > 0)]
                    right_target_poses = np.array(target_poses)[np.where(np.array(target_poses)[:, 0] < 0)]

                    if "brow_" in morph_name:
                        # 眉
                        # デフォルトの移動量（とりあえず適当に）
                        offset_distance = 0.2
                        # 眉は目との距離
                        eyeline_material_vertices = None
                        for mat_name, mat_vert in model.material_vertices.items():
                            if "_FaceEyeline_" in mat_name:
                                eyeline_material_vertices = mat_vert
                                break
                        # 目の位置データ
                        eyeline_poses = []
                        if eyeline_material_vertices:
                            for vidx in eyeline_material_vertices:
                                vertex = model.vertex_dict[vidx]
                                eyeline_poses.append(vertex.position.data())
                            if target_poses and eyeline_poses:
                                max_target_pos = np.max(target_poses, axis=0)
                                max_eyeline_pos = np.max(eyeline_poses, axis=0)
                                diff_pos = max_target_pos - max_eyeline_pos
                                offset_distance = diff_pos[1] * 0.6

                        for vidx in target_material_vertices:
                            vertex = model.vertex_dict[vidx]
                            if (
                                ("_R" == morph_name[-2:] and vertex.position.x() < 0)
                                or ("_L" == morph_name[-2:] and vertex.position.x() > 0)
                                or ("_R" != morph_name[-2:] and "_L" != morph_name[-2:])
                            ):
                                if "_Below" in morph_name:
                                    morph_offset = VertexMorphOffset(vertex.index, MVector3D(0, -offset_distance, 0))
                                if "_Abobe" in morph_name:
                                    morph_offset = VertexMorphOffset(vertex.index, MVector3D(0, offset_distance, 0))
                                if "_Left" in morph_name:
                                    morph_offset = VertexMorphOffset(vertex.index, MVector3D(offset_distance, 0, 0))
                                if "_Right" in morph_name:
                                    morph_offset = VertexMorphOffset(vertex.index, MVector3D(-offset_distance, 0, 0))
                                if "_Front" in morph_name:
                                    morph_offset = VertexMorphOffset(vertex.index, MVector3D(0, 0, -offset_distance))
                                if "_Front" not in morph_name:
                                    # 眉前以外はZ方向に補正する
                                    morphed_pos = (vertex.position + morph_offset.position_offset).data()

                                    # 面ごとの頂点との距離
                                    face_index_distances = np.linalg.norm(
                                        (face_material_index_vertices[:, :, :2] - morphed_pos[:2]),
                                        ord=2,
                                        axis=2,
                                    )
                                    nearest_face_index_vertices = face_material_index_vertices[
                                        np.argmin(np.sum(face_index_distances, axis=1))
                                    ]

                                    # 面垂線
                                    v1 = nearest_face_index_vertices[1] - nearest_face_index_vertices[0]
                                    v2 = nearest_face_index_vertices[2] - nearest_face_index_vertices[1]
                                    surface_vector = MVector3D.crossProduct(MVector3D(v1), MVector3D(v2))
                                    surface_normal = surface_vector.normalized()

                                    morphed_nearest_pos = calc_intersect_point(
                                        morphed_pos + np.array([0, 0, -1000]),
                                        morphed_pos + np.array([0, 0, 1000]),
                                        np.mean(nearest_face_index_vertices, axis=0),
                                        surface_normal.data(),
                                    )

                                    # Z方向の補正
                                    morph_offset.position_offset.setZ(morphed_nearest_pos[2] - morphed_pos[2] - 0.02)
                                target_offset.append(morph_offset)

                    elif "eye_Small" in morph_name:
                        if not ("Fcl_EYE_Surprised_R" in target_morphs and "Fcl_EYE_Surprised_L" in target_morphs):
                            logger.warning(
                                "Fcl_EYE_Surprised モーフがなかったため、瞳小モーフ生成をスルーします", decoration=MLogger.DECORATION_BOX
                            )
                            continue
                        # 瞳小
                        base_morph = (
                            target_morphs["Fcl_EYE_Surprised_R"]
                            if "eye_Small_R" == morph_name
                            else target_morphs["Fcl_EYE_Surprised_L"]
                        )
                        for base_offset in base_morph.offsets:
                            # びっくりの目部分だけ抜き出す
                            if base_offset.vertex_index in target_material_vertices:
                                target_offset.append(copy.deepcopy(base_offset))
                    elif "eye_Big" in morph_name:
                        if not ("Fcl_EYE_Surprised_R" in target_morphs and "Fcl_EYE_Surprised_L" in target_morphs):
                            logger.warning(
                                "Fcl_EYE_Surprised モーフがなかったため、瞳大モーフ生成をスルーします", decoration=MLogger.DECORATION_BOX
                            )
                            continue
                        # 瞳大
                        base_morph = (
                            target_morphs["Fcl_EYE_Surprised_R"]
                            if "eye_Big_R" == morph_name
                            else target_morphs["Fcl_EYE_Surprised_L"]
                        )
                        for base_offset in base_morph.offsets:
                            # びっくりの目部分だけ抜き出して大きさ反転
                            if base_offset.vertex_index in target_material_vertices:
                                target_offset.append(
                                    VertexMorphOffset(base_offset.vertex_index, base_offset.position_offset * -1)
                                )
                    elif "eye_Hide_Vertex" in morph_name:
                        if not ("Fcl_EYE_Close" in target_morphs):
                            logger.warning(
                                "Fcl_EYE_Close モーフがなかったため、目隠し頂点モーフ生成をスルーします", decoration=MLogger.DECORATION_BOX
                            )
                            continue
                        for base_offset in target_morphs["Fcl_EYE_Close"].offsets:
                            vertex = model.vertex_dict[base_offset.vertex_index]
                            if base_offset.vertex_index in hide_material_vertices:
                                # アイラインは両目ボーンの位置に合わせる
                                target_offset.append(
                                    VertexMorphOffset(vertex.index, model.bones["両目"].position - vertex.position)
                                )
                            else:
                                # その他はそのまままばたきの変動
                                target_offset.append(
                                    VertexMorphOffset(
                                        vertex.index, base_offset.position_offset * MVector3D(1, 1.05, 1)
                                    )
                                )

                        for vidx in target_material_vertices:
                            vertex = model.vertex_dict[vidx]
                            morph_offset = VertexMorphOffset(
                                vertex.index,
                                MVector3D(),
                            )

                            # 顔頂点を左右に分ける
                            face_target_material_index_vertices = (
                                face_left_close_index_vertices
                                if np.sign(vertex.position.x()) > 0
                                else face_right_close_index_vertices
                            )
                            # 白目材質を真円に広げる
                            vertex_target_poses = (
                                left_target_poses if np.sign(vertex.position.x()) > 0 else right_target_poses
                            )
                            mean_target_pos = np.mean(vertex_target_poses, axis=0)
                            min_target_pos = np.min(vertex_target_poses, axis=0)
                            max_target_pos = np.max(vertex_target_poses, axis=0)
                            target_diff = (max_target_pos - min_target_pos)[:2]
                            if target_diff[0] > target_diff[1]:
                                # 切れ長の目
                                morph_offset.position_offset.setX(
                                    abs(
                                        (
                                            abs(vertex.position.x() - mean_target_pos[0])
                                            * target_diff[1]
                                            / target_diff[0]
                                        )
                                        - abs(vertex.position.x() - mean_target_pos[0])
                                    )
                                    * np.sign(mean_target_pos[0] - vertex.position.x())
                                )
                            else:
                                # 縦長の目
                                morph_offset.position_offset.setY(
                                    abs(
                                        (
                                            abs(vertex.position.y() - mean_target_pos[1])
                                            * target_diff[0]
                                            / target_diff[1]
                                        )
                                        - abs(vertex.position.y() - mean_target_pos[1])
                                    )
                                    * np.sign(mean_target_pos[1] - vertex.position.y())
                                )

                            morph_offset.position_offset += (vertex.position - MVector3D(mean_target_pos)) * MVector3D(
                                0.1, 0.1, 0
                            )

                            morphed_pos = (vertex.position + morph_offset.position_offset).data()

                            # 面ごとの頂点との距離
                            face_index_distances = np.linalg.norm(
                                (face_target_material_index_vertices[:, :, :2] - morphed_pos[:2]),
                                ord=2,
                                axis=2,
                            )
                            nearest_face_index_vertices = face_target_material_index_vertices[
                                np.argmin(np.sum(face_index_distances, axis=1))
                            ]

                            # 面垂線
                            v1 = nearest_face_index_vertices[1] - nearest_face_index_vertices[0]
                            v2 = nearest_face_index_vertices[2] - nearest_face_index_vertices[1]
                            surface_vector = MVector3D.crossProduct(MVector3D(v1), MVector3D(v2))
                            surface_normal = surface_vector.normalized()

                            morphed_nearest_pos = calc_intersect_point(
                                morphed_pos + np.array([0, 0, -1000]),
                                morphed_pos + np.array([0, 0, 1000]),
                                np.mean(nearest_face_index_vertices, axis=0),
                                surface_normal.data(),
                            )

                            # Z方向の補正
                            morph_offset.position_offset.setZ(morphed_nearest_pos[2] - morphed_pos[2] - 0.03)
                            # 白目部分を前に出す
                            target_offset.append(morph_offset)

                    if target_offset:
                        morph = Morph(morph_pair["name"], morph_name, morph_pair["panel"], 1)
                        morph.index = len(target_morphs)
                        active_target_offset = []
                        for to in target_offset:
                            if (type(to) is VertexMorphOffset and to.position_offset != MVector3D()) or type(
                                to
                            ) is not VertexMorphOffset:
                                # 頂点モーフは値がある場合のみ適用
                                active_target_offset.append(to)
                        morph.offsets = active_target_offset

                        target_morphs[morph_name] = morph
            elif "material" in morph_pair:
                # 材質モーフ
                morph = None
                for material_index, material in enumerate(model.materials.values()):
                    if morph_pair["material"] in model.textures[material.texture_index]:
                        # 材質名が含まれている場合、対象
                        if not morph:
                            morph = Morph(morph_pair["name"], morph_name, morph_pair["panel"], 8)
                            morph.index = len(target_morphs)

                        morph.offsets.append(
                            MaterialMorphData(
                                material_index,
                                1,
                                MVector4D(0, 0, 0, 1),
                                MVector3D(),
                                0,
                                MVector3D(),
                                MVector4D(),
                                0,
                                MVector4D(),
                                MVector4D(),
                                MVector4D(),
                            )
                        )

                    elif "hides" in morph_pair and np.count_nonzero(
                        [material.name.endswith(hide_morph) for hide_morph in morph_pair["hides"]]
                    ):
                        # 隠す材質名が含まれている場合、対象
                        if not morph:
                            morph = Morph(morph_pair["name"], morph_name, morph_pair["panel"], 8)
                            morph.index = len(target_morphs)

                        morph.offsets.append(
                            MaterialMorphData(
                                material_index,
                                0,
                                MVector4D(0, 0, 0, 0),
                                MVector3D(),
                                0,
                                MVector3D(),
                                MVector4D(),
                                0,
                                MVector4D(),
                                MVector4D(),
                                MVector4D(),
                            )
                        )

                if morph:
                    target_morphs[morph_name] = morph
            elif "edge" in morph_pair:
                # エッジOFF
                morph = None
                for material_index, material in enumerate(model.materials.values()):
                    if (material.flag & 0x10) != 0:
                        if not morph:
                            morph = Morph(morph_pair["name"], morph_name, morph_pair["panel"], 8)
                            morph.index = len(target_morphs)

                        # エッジONの場合、OFFにするモーフ追加
                        morph.offsets.append(
                            MaterialMorphData(
                                material_index,
                                0,
                                MVector4D(1, 1, 1, 1),
                                MVector3D(1, 1, 1),
                                1,
                                MVector3D(1, 1, 1),
                                # エッジのサイズと透明度だけ0
                                MVector4D(1, 1, 1, 0),
                                0,
                                MVector4D(1, 1, 1, 1),
                                MVector4D(1, 1, 1, 1),
                                MVector4D(1, 1, 1, 1),
                            )
                        )
                    elif material.name.endswith("_エッジ"):
                        # エッジ材質の場合、全部OFF
                        if not morph:
                            morph = Morph(morph_pair["name"], morph_name, morph_pair["panel"], 8)
                            morph.index = len(target_morphs)

                        morph.offsets.append(
                            MaterialMorphData(
                                material_index,
                                0,
                                MVector4D(0, 0, 0, 0),
                                MVector3D(),
                                0,
                                MVector3D(),
                                MVector4D(),
                                0,
                                MVector4D(),
                                MVector4D(),
                                MVector4D(),
                            )
                        )

                if morph:
                    target_morphs[morph_name] = morph
            elif "bone" in morph_pair:
                morph = Morph(morph_pair["name"], morph_name, morph_pair["panel"], 2)
                morph.index = len(target_morphs) + 100
                for bname, move_ratio, rotate_ratio in zip(
                    morph_pair["bone"],
                    morph_pair["move_ratios"],
                    morph_pair["rotate_ratios"],
                ):
                    morph.offsets.append(BoneMorphData(model.bones[bname].index, move_ratio, rotate_ratio))
                target_morphs[morph_name] = morph
            else:
                if morph_name in target_morphs:
                    morph = target_morphs[morph_name]
                    morph.name = morph_pair["name"]
                    morph.panel = morph_pair["panel"]

        logger.info("-- -- 拡張モーフ調整: %s個目", sidx)

        target_morph_indexes = {}
        for morph_name in MORPH_PAIRS.keys():
            if morph_name not in target_morphs:
                continue
            morph = target_morphs[morph_name]
            # モーフINDEX新旧比較
            target_morph_indexes[morph.index] = len(model.org_morphs)
            morph.index = len(model.org_morphs)
            model.org_morphs[morph.name] = morph

        for morph in model.org_morphs.values():
            if morph.morph_type == 0:
                # グループモーフの場合、新旧入替
                for offset in morph.offsets:
                    old_idx = offset.morph_index
                    offset.morph_index = target_morph_indexes[old_idx]
            elif morph.morph_type == 1:
                # 頂点モーフの場合、動いてないのは削除
                movable_offsets = []
                for offset in morph.offsets:
                    if offset.position_offset != MVector3D():
                        movable_offsets.append(offset)
                morph.offsets = movable_offsets

            if "材質" not in morph.name and "頂点" not in morph.name and "ボーン" not in morph.name:
                model.display_slots["表情"].references.append((1, morph.index))

        for morph_name, morph in vertex_morphs.items():
            # 定義外モーフがあれば一応取り込む（表示枠には追加しない）
            if morph_name in MORPH_PAIRS.keys() and (
                morph_name in target_morphs or morph.name in target_morphs or morph.english_name in target_morphs
            ):
                continue
            target_morph_indexes[morph.index] = len(model.org_morphs)
            morph.index = len(model.org_morphs)
            model.org_morphs[morph.name] = morph

        logger.info("-- グループモーフデータ解析")

        return model

    def reconvert_bone(self, model: PmxModel):
        # 指先端の位置を計算して配置
        finger_dict = {
            "左親指２": {"vertices": [], "direction": -1, "edge_name": "左親指先"},
            "左人指３": {"vertices": [], "direction": -1, "edge_name": "左人指先"},
            "左中指３": {"vertices": [], "direction": -1, "edge_name": "左中指先"},
            "左薬指３": {"vertices": [], "direction": -1, "edge_name": "左薬指先"},
            "左小指３": {"vertices": [], "direction": -1, "edge_name": "左小指先"},
            "右親指２": {"vertices": [], "direction": 1, "edge_name": "右親指先"},
            "右人指３": {"vertices": [], "direction": 1, "edge_name": "右人指先"},
            "右中指３": {"vertices": [], "direction": 1, "edge_name": "右中指先"},
            "右薬指３": {"vertices": [], "direction": 1, "edge_name": "右薬指先"},
            "右小指３": {"vertices": [], "direction": 1, "edge_name": "右小指先"},
        }
        # つま先の位置を計算して配置
        toe_dict = {
            "左足先EX": {"vertices": [], "edge_name": "左つま先", "ik_name": "左つま先ＩＫ"},
            "右足先EX": {"vertices": [], "edge_name": "右つま先", "ik_name": "右つま先ＩＫ"},
        }

        for vertex_idx, vertex in model.vertex_dict.items():
            if type(vertex.deform) is Bdef1:
                # 指先に相当する頂点位置をリスト化
                for finger_name in finger_dict.keys():
                    if model.bones[finger_name].index == vertex.deform.index0:
                        finger_dict[finger_name]["vertices"].append(vertex.position)
                # つま先に相当する頂点位置をリスト化
                for toe_name in toe_dict.keys():
                    if model.bones[toe_name].index == vertex.deform.index0:
                        toe_dict[toe_name]["vertices"].append(vertex.position)

        for finger_name, finger_param in finger_dict.items():
            if len(finger_param["vertices"]) > 0:
                # 末端頂点の位置を指先ボーンの位置として割り当て
                finger_vertices = sorted(finger_param["vertices"], key=lambda v: v.x() * finger_param["direction"])
                edge_vertex_pos = finger_vertices[0]
                model.bones[finger_param["edge_name"]].position = edge_vertex_pos

        for toe_name, toe_param in toe_dict.items():
            if len(toe_param["vertices"]) > 0:
                # 末端頂点の位置をつま先ボーンの位置として割り当て
                toe_vertices = sorted(toe_param["vertices"], key=lambda v: v.z())
                edge_vertex_pos = toe_vertices[0].copy()
                # Yは0に固定
                edge_vertex_pos.setY(0)
                model.bones[toe_param["edge_name"]].position = edge_vertex_pos
                model.bones[toe_param["ik_name"]].position = edge_vertex_pos

        for leg_bone_name in ["腰キャンセル左", "腰キャンセル右", "左足", "右足", "左足D", "右足D"]:
            if leg_bone_name in model.bones:
                model.bones[leg_bone_name].position.setZ(model.bones[leg_bone_name].position.z() + 0.1)

        for knee_bone_name in ["左ひざ", "右ひざ", "左ひざD", "右ひざD"]:
            if knee_bone_name in model.bones:
                model.bones[knee_bone_name].position.setZ(model.bones[knee_bone_name].position.z() - 0.1)

        # 体幹を中心に揃える
        for trunk_bone_name in ["全ての親", "センター", "グルーブ", "腰", "下半身", "上半身", "上半身2", "上半身3", "首", "頭", "両目"]:
            model.bones[trunk_bone_name].position.setX(0)

        # 左右ボーンを線対称に揃える
        for left_bone_name, left_bone in model.bones.items():
            right_bone_name = f"右{left_bone_name[1:]}"
            if "左" == left_bone_name[0] and right_bone_name in model.bones:
                right_bone = model.bones[right_bone_name]
                mean_position = MVector3D(
                    np.mean([abs(left_bone.position.x()), abs(right_bone.position.x())]),
                    np.mean([left_bone.position.y(), right_bone.position.y()]),
                    np.mean([left_bone.position.z(), right_bone.position.z()]),
                )
                left_bone.position = MVector3D(
                    mean_position.x() * np.sign(left_bone.position.x()), mean_position.y(), mean_position.z()
                )
                right_bone.position = MVector3D(
                    mean_position.x() * np.sign(right_bone.position.x()), mean_position.y(), mean_position.z()
                )

        highlight_material_name = None
        for (mat_name, material) in model.materials.items():
            if "EyeHighlight" in material.name:
                highlight_material_name = mat_name
                break

        if highlight_material_name:
            # ハイライトボーンに置き換え
            for eye_bone_name, highlight_bone_name in [("左目", "左目光"), ("右目", "右目光")]:
                if (
                    highlight_material_name not in model.material_vertices
                    or model.bones[eye_bone_name].index not in model.vertices
                ):
                    continue

                model.bones[highlight_bone_name].position = model.bones[eye_bone_name].position.copy()

                highlight_vidxs = list(
                    set(model.material_vertices[highlight_material_name])
                    & set(model.vertices[model.bones[eye_bone_name].index])
                )

                for vidx in highlight_vidxs:
                    v = model.vertex_dict[vidx]

                    if model.bones[eye_bone_name].index in v.deform.get_idx_list():
                        if type(v.deform) is Bdef1:
                            v.deform.index0 = model.bones[highlight_bone_name].index
                        elif type(v.deform) is Bdef2:
                            if v.deform.index0 == model.bones[eye_bone_name].index:
                                v.deform.index0 = model.bones[highlight_bone_name].index
                            if v.deform.index1 == model.bones[eye_bone_name].index:
                                v.deform.index1 = model.bones[highlight_bone_name].index
                        elif type(v.deform) is Bdef4:
                            if v.deform.index0 == model.bones[eye_bone_name].index:
                                v.deform.index0 = model.bones[highlight_bone_name].index
                            if v.deform.index1 == model.bones[eye_bone_name].index:
                                v.deform.index1 = model.bones[highlight_bone_name].index
                            if v.deform.index2 == model.bones[eye_bone_name].index:
                                v.deform.index2 = model.bones[highlight_bone_name].index
                            if v.deform.index3 == model.bones[eye_bone_name].index:
                                v.deform.index3 = model.bones[highlight_bone_name].index

        tongue_material_name = None
        for (mat_name, material) in model.materials.items():
            if "FaceMouth" in material.name:
                tongue_material_name = mat_name
                break

        if (
            tongue_material_name
            and tongue_material_name in model.material_vertices
            and model.bones["頭"].index in model.vertices
            and "舌1" in model.bones
            and "舌2" in model.bones
            and "舌3" in model.bones
            and "舌4" in model.bones
        ):
            # 舌ボーンにウェイト置き換え
            tongue_vidxs = []
            tongue_vertices = []
            tongue_poses = []
            for vidx in model.material_vertices[tongue_material_name]:
                v = model.vertex_dict[vidx]
                if v.uv.x() >= 0.5 and v.uv.y() <= 0.5:
                    tongue_vidxs.append(vidx)
                    tongue_vertices.append(v)
                    tongue_poses.append(v.position.data())
                    v.deform = Bdef1(model.bones["舌1"].index)

            if tongue_poses and tongue_vertices and tongue_vidxs:
                # 舌ボーン位置配置
                tongue1_pos = np.max(tongue_poses, axis=0)
                model.bones["舌1"].position = MVector3D(0, tongue1_pos[1], tongue1_pos[2])
                tongue4_pos = np.min(tongue_poses, axis=0)
                model.bones["舌4"].position = MVector3D(0, tongue4_pos[1], tongue4_pos[2])
                model.bones["舌2"].position = model.bones["舌1"].position + (
                    (model.bones["舌4"].position - model.bones["舌1"].position) * 0.4
                )
                model.bones["舌3"].position = model.bones["舌1"].position + (
                    (model.bones["舌4"].position - model.bones["舌1"].position) * 0.7
                )

                for from_bone_name, to_bone_name in [("舌1", "舌2"), ("舌2", "舌3"), ("舌3", "舌4")]:
                    # ローカル軸
                    model.bones[from_bone_name].local_x_vector = (
                        model.bones[to_bone_name].position - model.bones[from_bone_name].position
                    ).normalized()
                    model.bones[from_bone_name].local_z_vector = MVector3D.crossProduct(
                        model.bones[from_bone_name].local_x_vector, MVector3D(0, -1, 0)
                    ).normalized()
                model.bones["舌4"].local_x_vector = model.bones["舌3"].local_x_vector.copy()
                model.bones["舌4"].local_z_vector = model.bones["舌3"].local_z_vector.copy()

                for vertex in tongue_vertices:
                    for from_bone_name, to_bone_name in [("舌1", "舌2"), ("舌2", "舌3"), ("舌3", "舌4")]:
                        tongue_distance = (
                            model.bones[to_bone_name].position.z() - model.bones[from_bone_name].position.z()
                        )
                        vertex_distance = vertex.position.z() - model.bones[from_bone_name].position.z()

                        if np.sign(tongue_distance) == np.sign(vertex_distance):
                            # 範囲内である場合、ウェイト分布
                            vertex.deform = Bdef2(
                                model.bones[to_bone_name].index,
                                model.bones[from_bone_name].index,
                                vertex_distance / tongue_distance,
                            )

                # 舌頂点モーフを削除
                for morph in model.org_morphs.values():
                    if morph.morph_type == 1:
                        # 頂点モーフの場合
                        without_tongue_offsets = []
                        for offset in morph.offsets:
                            if offset.vertex_index not in tongue_vidxs:
                                without_tongue_offsets.append(offset)
                        morph.offsets = without_tongue_offsets
            else:
                logger.warning("舌関連頂点が見つからなかったため、舌分離処理をスキップします", decoration=MLogger.DECORATION_BOX)

        logger.info("-- ボーンデータ調整終了")

        return model

    def convert_mesh(self, model: PmxModel, bone_name_dict: dict, tex_dir_path: str):
        if "meshes" not in model.json_data:
            logger.error("変換可能なメッシュ情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
            return None

        vertex_blocks = {}
        vertex_idx = 0

        for midx, mesh in enumerate(model.json_data["meshes"]):
            if "primitives" not in mesh:
                continue

            for pidx, primitive in enumerate(mesh["primitives"]):
                if (
                    "attributes" not in primitive
                    or "indices" not in primitive
                    or "material" not in primitive
                    or "JOINTS_0" not in primitive["attributes"]
                    or "NORMAL" not in primitive["attributes"]
                    or "POSITION" not in primitive["attributes"]
                    or "TEXCOORD_0" not in primitive["attributes"]
                    or "WEIGHTS_0" not in primitive["attributes"]
                ):
                    continue

                # 頂点ブロック
                vertex_key = f'{primitive["attributes"]["JOINTS_0"]}-{primitive["attributes"]["NORMAL"]}-{primitive["attributes"]["POSITION"]}-{primitive["attributes"]["TEXCOORD_0"]}-{primitive["attributes"]["WEIGHTS_0"]}'

                # 頂点データ
                if vertex_key not in vertex_blocks:
                    vertex_blocks[vertex_key] = {"vertices": [], "start": vertex_idx, "indices": [], "materials": []}

                    # 位置データ
                    positions = self.read_from_accessor(model, primitive["attributes"]["POSITION"])

                    # 法線データ
                    normals = self.read_from_accessor(model, primitive["attributes"]["NORMAL"])

                    # UVデータ
                    uvs = self.read_from_accessor(model, primitive["attributes"]["TEXCOORD_0"])

                    # ジョイントデータ(MMDのジョイントとは異なる)
                    if "JOINTS_0" in primitive["attributes"]:
                        joints = self.read_from_accessor(model, primitive["attributes"]["JOINTS_0"])
                    else:
                        joints = [MVector4D() for _ in range(len(positions))]

                    # ウェイトデータ
                    if "WEIGHTS_0" in primitive["attributes"]:
                        weights = self.read_from_accessor(model, primitive["attributes"]["WEIGHTS_0"])
                    else:
                        weights = [MVector4D() for _ in range(len(positions))]

                    # 対応するジョイントデータ
                    try:
                        skin_joints = model.json_data["skins"][
                            [s for s in model.json_data["nodes"] if "mesh" in s and s["mesh"] == midx][0]["skin"]
                        ]["joints"]
                    except Exception:
                        # 取れない場合はとりあえず空
                        skin_joints = []

                    if "extras" in primitive and "targetNames" in primitive["extras"] and "targets" in primitive:
                        for eidx, (extra, target) in enumerate(
                            zip(primitive["extras"]["targetNames"], primitive["targets"])
                        ):
                            # 位置データ
                            extra_positions = self.read_from_accessor(model, target["POSITION"])

                            # 法線データ
                            extra_normals = self.read_from_accessor(model, target["NORMAL"])

                            morph = Morph(extra, extra, MORPH_OTHER, 1)
                            morph.index = eidx

                            morph_vertex_idx = vertex_idx
                            for vidx, (eposition, enormal) in enumerate(zip(extra_positions, extra_normals)):
                                model_eposition = eposition * MIKU_METER * MVector3D(-1, 1, 1)

                                morph.offsets.append(VertexMorphOffset(morph_vertex_idx, model_eposition))
                                morph_vertex_idx += 1

                            model.org_morphs[extra] = morph

                    for position, normal, uv, joint, weight in zip(positions, normals, uvs, joints, weights):
                        model_position = position * MIKU_METER * MVector3D(-1, 1, 1)

                        # 有効なINDEX番号と実際のボーンINDEXを取得
                        joint_idxs, weight_values = self.get_deform_index(
                            vertex_idx, model, model_position, joint, skin_joints, weight, bone_name_dict
                        )
                        if len(joint_idxs) > 1:
                            if len(joint_idxs) == 2:
                                # ウェイトが2つの場合、Bdef2
                                deform = Bdef2(joint_idxs[0], joint_idxs[1], weight_values[0])
                            else:
                                # それ以上の場合、Bdef4
                                deform = Bdef4(
                                    joint_idxs[0],
                                    joint_idxs[1],
                                    joint_idxs[2],
                                    joint_idxs[3],
                                    weight_values[0],
                                    weight_values[1],
                                    weight_values[2],
                                    weight_values[3],
                                )
                        elif len(joint_idxs) == 1:
                            # ウェイトが1つのみの場合、Bdef1
                            deform = Bdef1(joint_idxs[0])
                        else:
                            # とりあえず除外
                            deform = Bdef1(0)

                        vertex = Vertex(
                            vertex_idx,
                            model_position,
                            (normal * MVector3D(-1, 1, 1)).normalized(),
                            uv,
                            None,
                            deform,
                            1,
                        )

                        model.vertex_dict[vertex_idx] = vertex
                        # verticesはとりあえずボーンINDEXで管理
                        for bidx in deform.get_idx_list():
                            if bidx not in model.vertices:
                                model.vertices[bidx] = []
                            model.vertices[bidx].append(vertex_idx)
                        vertex_blocks[vertex_key]["vertices"].append(vertex_idx)
                        vertex_idx += 1

                    logger.info("-- 頂点データ解析[%s]", vertex_key)

                vertex_blocks[vertex_key]["indices"].append(primitive["indices"])
                vertex_blocks[vertex_key]["materials"].append(primitive["material"])

        hair_regexp = r"((N\d+_\d+_Hair_\d+)_HAIR)"
        tex_regexp = r"_(\d+)"

        indices_by_materials = {}
        materials_by_type = {}
        registed_material_names = []

        for vertex_key, vertex_dict in vertex_blocks.items():
            start_vidx = vertex_dict["start"]
            indices = vertex_dict["indices"]
            materials = vertex_dict["materials"]

            for index_accessor, material_accessor in zip(indices, materials):
                # 材質データ ---------------
                vrm_material = model.json_data["materials"][material_accessor]
                material_name = vrm_material["name"]

                # 材質順番を決める
                material_key = vrm_material["alphaMode"]
                for mkey_type in [
                    "_FaceMouth",
                    "_Face_",
                    "_Body_",
                    "_Hair_",
                    "_HairBack_",
                    "_FaceBrow",
                    "_FaceEyeline",
                    "_FaceEyelash",
                    "_EyeWhite",
                    "_EyeIris",
                    "_EyeHighlight",
                    "Lens",
                    "Accessory",
                ]:
                    if mkey_type in material_name:
                        material_key = mkey_type
                        break

                if material_key not in materials_by_type:
                    materials_by_type[material_key] = {}

                if material_name not in materials_by_type[material_key]:
                    # VRMの材質拡張情報
                    material_ext = [
                        m
                        for m in model.json_data["extensions"]["VRM"]["materialProperties"]
                        if m["name"] == material_name
                    ][0]
                    # 非透過度
                    # 拡散色
                    diffuse_color_data = vrm_material["pbrMetallicRoughness"]["baseColorFactor"]
                    alpha = diffuse_color_data[3]
                    specular_factor = 0
                    # diffuse_color = MVector3D(*diffuse_color_data[:3])
                    # # 反射色
                    # if "emissiveFactor" in vrm_material:
                    #     specular_color_data = vrm_material["emissiveFactor"]
                    #     specular_color = MVector3D(*specular_color_data[:3])
                    # else:
                    #     specular_color = MVector3D()
                    # # 環境色
                    # if "vectorProperties" in material_ext and "_ShadeColor" in material_ext["vectorProperties"]:
                    #     ambient_color = MVector3D(*material_ext["vectorProperties"]["_ShadeColor"][:3])
                    # else:
                    #     ambient_color = diffuse_color / 2
                    # 拡散色・反射色・環境色は固定とする
                    diffuse_color = MVector3D(1, 1, 1)
                    specular_color = MVector3D()
                    ambient_color = diffuse_color * 0.5
                    # 0x02:地面影, 0x04:セルフシャドウマップへの描画, 0x08:セルフシャドウの描画
                    flag = 0x02 | 0x04 | 0x08
                    if vrm_material["doubleSided"]:
                        # 両面描画
                        flag |= 0x01
                    edge_color = MVector4D(*material_ext["vectorProperties"]["_OutlineColor"])
                    edge_size = material_ext["floatProperties"]["_OutlineWidth"] * MIKU_METER
                    # エッジ描画は透明ポリの裏面がエッジ色で塗りつぶされてしまうので、ここでフラグはONにしない

                    sphere_mode = 2
                    texture_index = 0
                    sphere_texture_index = 0

                    # 0番目は空テクスチャなので+1で設定
                    m = re.search(hair_regexp, material_name)
                    if m is not None:
                        # 髪材質の場合、合成
                        hair_img_name = os.path.basename(
                            model.textures[material_ext["textureProperties"]["_MainTex"] + 1]
                        )
                        hm = re.search(tex_regexp, hair_img_name)
                        hair_img_number = -1
                        if hm is not None:
                            hair_img_number = int(hm.groups()[0])

                        # 髪テクスチャはそのまま保持
                        model.textures.append(os.path.join("tex", hair_img_name))
                        texture_index = len(model.textures) - 1

                        # スフィアファイルをコピー
                        hair_spe_name = f"hair_sphere_{hair_img_number}.png"
                        shutil.copy(
                            MFileUtils.resource_path(f"src/resources/hair_sphere.png"),
                            os.path.join(tex_dir_path, hair_spe_name),
                        )
                        model.textures.append(os.path.join("tex", hair_spe_name))
                        sphere_texture_index = len(model.textures) - 1

                        spe_img = Image.open(os.path.join(tex_dir_path, hair_spe_name)).convert("RGBA")
                        spe_ary = np.array(spe_img)

                        # 反射色の画像
                        if "emissiveFactor" in vrm_material:
                            emissive_ary = np.array(vrm_material["emissiveFactor"])
                            emissive_ary = np.append(emissive_ary, 1)
                        else:
                            # なかった場合には仮に明るめの色を入れておく
                            logger.warning("髪の反射色がVrmデータになかったため、仮に白色を差し込みます", decoration=MLogger.DECORATION_BOX)
                            emissive_ary = np.array([0.9, 0.9, 0.9, 1])

                        # 反射色だけの画像生成
                        emissive_img = Image.fromarray(
                            np.tile(emissive_ary * 255, (spe_ary.shape[0], spe_ary.shape[1], 1)).astype(np.uint8),
                            mode="RGBA",
                        )
                        # 乗算して保存
                        hair_emissive_img = ImageChops.multiply(spe_img, emissive_img)
                        hair_emissive_img.save(os.path.join(tex_dir_path, hair_spe_name))

                        # ---------
                        # 髪の毛にハイライトを焼き込んだ画像も作るだけ作っておく
                        hair_spe_only_name = f"_{(hair_img_number + 1):02d}.png"
                        hair_blend_name = f"_{hair_img_number:02d}_blend.png"

                        if os.path.exists(os.path.join(tex_dir_path, hair_img_name)) and os.path.exists(
                            os.path.join(tex_dir_path, hair_spe_only_name)
                        ):
                            # スペキュラファイルがある場合
                            hair_img = Image.open(os.path.join(tex_dir_path, hair_img_name)).convert("RGBA")
                            hair_ary = np.array(hair_img)

                            spe_img = Image.open(os.path.join(tex_dir_path, hair_spe_only_name)).convert("RGBA")
                            spe_ary = np.array(spe_img)

                            # 拡散色の画像
                            diffuse_ary = np.array(material_ext["vectorProperties"]["_Color"])
                            diffuse_img = Image.fromarray(
                                np.tile(diffuse_ary * 255, (hair_ary.shape[0], hair_ary.shape[1], 1)).astype(np.uint8),
                                mode="RGBA",
                            )
                            hair_diffuse_img = ImageChops.multiply(hair_img, diffuse_img)

                            # 反射色の画像
                            if "emissiveFactor" in vrm_material:
                                emissive_ary = np.array(vrm_material["emissiveFactor"])
                                emissive_ary = np.append(emissive_ary, 1)
                            else:
                                # なかった場合には仮に明るめの色を入れておく
                                logger.warning("髪の反射色がVrmデータになかったため、仮に白色を差し込みます", decoration=MLogger.DECORATION_BOX)
                                emissive_ary = np.array([0.9, 0.9, 0.9, 1])
                            emissive_img = Image.fromarray(
                                np.tile(emissive_ary * 255, (spe_ary.shape[0], spe_ary.shape[1], 1)).astype(np.uint8),
                                mode="RGBA",
                            )
                            # 乗算
                            hair_emissive_img = ImageChops.multiply(spe_img, emissive_img)
                            # スクリーン
                            dest_img = ImageChops.screen(hair_diffuse_img, hair_emissive_img)
                            dest_img.save(os.path.join(tex_dir_path, hair_blend_name))
                    else:
                        if diffuse_color_data[:] != [1, 1, 1, 1]:
                            # 基本色が設定されている場合、加算しておく
                            logger.warning(
                                "基本色が白ではないため、加算合成します。 材質名: %s", material_name, decoration=MLogger.DECORATION_BOX
                            )

                            base_img_name = os.path.basename(
                                model.textures[material_ext["textureProperties"]["_MainTex"] + 1]
                            )
                            bm = re.search(tex_regexp, base_img_name)
                            base_img_number = -1
                            if bm is not None:
                                base_img_number = int(bm.groups()[0])
                            base_blend_name = f"_{base_img_number:02d}_blend.png"

                            base_img = Image.open(os.path.join(tex_dir_path, base_img_name)).convert("RGBA")
                            base_ary = np.array(base_img)

                            add_img = Image.fromarray(
                                np.tile(
                                    np.array(diffuse_color_data) * 255, (base_ary.shape[0], base_ary.shape[1], 1)
                                ).astype(np.uint8),
                                mode="RGBA",
                            )
                            base_add_img = ImageChops.multiply(base_img, add_img)
                            base_add_img.save(os.path.join(tex_dir_path, base_blend_name))

                            model.textures.append(os.path.join("tex", base_blend_name))
                            texture_index = len(model.textures) - 1
                        else:
                            # そのまま出力
                            texture_index = material_ext["textureProperties"]["_MainTex"] + 1

                        sphere_texture_index = 0
                        if "_SphereAdd" in material_ext["textureProperties"]:
                            sphere_texture_index = material_ext["textureProperties"]["_SphereAdd"] + 1
                            # 加算スフィア
                            sphere_mode = 2

                    if "vectorProperties" in material_ext and "_ShadeColor" in material_ext["vectorProperties"]:
                        toon_sharing_flag = 0
                        if material_ext["textureProperties"]["_MainTex"] < len(model.json_data["images"]):
                            toon_img_name = f'{model.json_data["images"][material_ext["textureProperties"]["_MainTex"]]["name"]}_Toon.bmp'
                        else:
                            toon_img_name = f"{material_name}_Toon.bmp"

                        toon_light_ary = np.tile(np.array([255, 255, 255, 255]), (24, 32, 1))
                        toon_shadow_ary = np.tile(
                            np.array(material_ext["vectorProperties"]["_ShadeColor"]) * 255, (8, 32, 1)
                        )
                        toon_ary = np.concatenate((toon_light_ary, toon_shadow_ary), axis=0)
                        toon_img = Image.fromarray(toon_ary.astype(np.uint8))

                        toon_img.save(os.path.join(tex_dir_path, toon_img_name))
                        model.textures.append(os.path.join("tex", toon_img_name))
                        # 最後に追加したテクスチャをINDEXとして設定
                        toon_texture_index = len(model.textures) - 1
                    else:
                        toon_sharing_flag = 1
                        toon_texture_index = 1

                    target_material_names = material_name.split(" ")
                    material_names = target_material_names[0].split("_")
                    material_name_ja = "_".join(material_names[-4:-2])
                    if material_names[-1].isdecimal():
                        # 末尾が数字である場合、末尾を入れる
                        material_name_ja = "_".join([material_names[-5], material_names[-4], material_names[-1]])
                    if material_name_ja in registed_material_names:
                        logger.warning(
                            "既に同じ材質名が登録されているため、元の材質名のまま登録します 変換材質名: %s 元材質名: %s",
                            material_name_ja,
                            material_name,
                            decoration=MLogger.DECORATION_BOX,
                        )
                        material_name_ja = target_material_names[0]

                    # 材質日本語名は部分抽出
                    material = Material(
                        material_name_ja,
                        material_name,
                        diffuse_color,
                        alpha,
                        specular_factor,
                        specular_color,
                        ambient_color,
                        flag,
                        edge_color,
                        edge_size,
                        texture_index,
                        sphere_texture_index,
                        sphere_mode,
                        toon_sharing_flag,
                        toon_texture_index,
                        "",
                        0,
                    )
                    registed_material_names.append(material_name_ja)
                    materials_by_type[material_key][material_name] = material
                    indices_by_materials[material_name] = {}
                else:
                    material = materials_by_type[material_key][material_name]

                # 面データ ---------------
                indices = self.read_from_accessor(model, index_accessor)
                indices_by_materials[material_name][index_accessor] = (np.array(indices) + start_vidx).tolist()
                material.vertex_count += len(indices)

                logger.info("-- 面・材質データ解析[%s-%s]", index_accessor, material_accessor)

        # 材質を透過順に並べ替て設定
        index_idx = 0
        for material_type in [
            "OPAQUE",
            "_Body_",
            "_FaceMouth",
            "_Face_",
            "_HairBack_",
            "_Hair_",
            "_FaceBrow",
            "_FaceEyeline",
            "_FaceEyelash",
            "_EyeWhite",
            "_EyeIris",
            "_EyeHighlight",
            "Accessory",
            "MASK",
            "BLEND",
            "Lens",
        ]:
            if material_type in materials_by_type:
                for material_name, material in materials_by_type[material_type].items():
                    is_append_edge_material = False
                    is_edge_flg_on = False

                    if material_type in ["_Body_", "_Face_", "_Hair_", "Accessory"] and material.edge_size > 0:
                        # エッジ設定可能材質でエッジサイズが指定されている場合、エッジON
                        is_edge_flg_on = True

                    elif material_type in ["MASK", "BLEND"] and material.edge_size > 0 and material.texture_index > 0:
                        # 一般材質（服系）に透明部分がある、かつエッジがある、かつテクスチャがある場合
                        is_edge_flg_on = True

                        # 該当テクスチャを読み込み
                        tex_ary = np.array(
                            Image.open(
                                os.path.join(tex_dir_path, os.path.basename(model.textures[material.texture_index]))
                            ).convert("RGBA")
                        )
                        for index_accessor, indices in indices_by_materials[material_name].items():
                            for vidx in indices:
                                v = model.vertex_dict[vidx]
                                # テクスチャのuvの位置を取得
                                uv_idx = (tex_ary.shape[:2] * np.array([v.uv.y(), v.uv.x()])).astype(np.int)
                                if (
                                    uv_idx[0] < tex_ary.shape[0]
                                    and uv_idx[1] < tex_ary.shape[1]
                                    and tex_ary[uv_idx[0], uv_idx[1]][3] < 255
                                ):
                                    logger.info(
                                        "テクスチャの透明部分がUVに含まれているため、エッジ材質を作成します 材質名: %s",
                                        material.name,
                                        decoration=MLogger.DECORATION_BOX,
                                    )
                                    is_edge_flg_on = False
                                    is_append_edge_material = True
                                    break
                            if is_append_edge_material:
                                break

                    if is_append_edge_material:
                        # 元材質の両面描画をOFFにする(裏もOFFなのでここのタイミング)
                        material.flag -= 0x01

                        back_material = copy.deepcopy(material)
                        back_material.name = f"{material.name}_裏"
                        back_material.english_name = f"{material_name}_Back"
                        model.materials[back_material.name] = back_material
                        model.material_vertices[back_material.name] = []

                        # エッジ材質を追加する場合
                        for index_accessor, indices in indices_by_materials[material_name].items():
                            for v0_idx, v1_idx, v2_idx in zip(indices[:-2:3], indices[1:-1:3], indices[2::3]):

                                # 一般材質（服系）に透明部分がある、かつエッジがある場合、裏面用に頂点を複製する
                                v0 = model.vertex_dict[v0_idx].copy()
                                v0.index = len(model.vertex_dict)
                                model.vertex_dict[v0.index] = v0

                                v1 = model.vertex_dict[v1_idx].copy()
                                v1.index = len(model.vertex_dict)
                                model.vertex_dict[v1.index] = v1

                                v2 = model.vertex_dict[v2_idx].copy()
                                v2.index = len(model.vertex_dict)
                                model.vertex_dict[v2.index] = v2

                                for v in [v0, v1, v2]:
                                    # 法線を反転させる
                                    v.normal *= -1
                                    v.normal.normalize()

                                # 裏面として貼り付けるので、INDEXの順番はそのまま
                                model.indices[index_idx] = [v0.index, v1.index, v2.index]
                                index_idx += 1

                                if v0.index not in model.material_vertices[back_material.name]:
                                    model.material_vertices[back_material.name].append(v0.index)

                                if v1.index not in model.material_vertices[back_material.name]:
                                    model.material_vertices[back_material.name].append(v1.index)

                                if v2.index not in model.material_vertices[back_material.name]:
                                    model.material_vertices[back_material.name].append(v2.index)

                    # 本来の材質
                    if is_edge_flg_on:
                        # エッジ材質を追加しない場合、エッジFLG=ON
                        material.flag |= 0x10

                    model.materials[material_name] = material
                    model.material_vertices[material_name] = []
                    model.material_indices[material_name] = []
                    for index_accessor, indices in indices_by_materials[material_name].items():
                        for v0_idx, v1_idx, v2_idx in zip(indices[:-2:3], indices[1:-1:3], indices[2::3]):
                            model.material_indices[material_name].append(index_idx)

                            # 面の貼り方がPMXは逆
                            model.indices[index_idx] = [v2_idx, v1_idx, v0_idx]
                            index_idx += 1

                            if v0_idx not in model.material_vertices[material_name]:
                                model.material_vertices[material_name].append(v0_idx)

                            if v1_idx not in model.material_vertices[material_name]:
                                model.material_vertices[material_name].append(v1_idx)

                            if v2_idx not in model.material_vertices[material_name]:
                                model.material_vertices[material_name].append(v2_idx)

                    append_materials = None
                    if material_type == "_EyeIris":
                        append_materials = [("eye_star", "星", "Star"), ("eye_heart", "ハート", "Heart")]
                    elif "_Face_" in material.english_name:
                        append_materials = [("cheek_dye", "頬染め", "Cheek_dye")]
                    elif material_type == "_EyeWhite":
                        append_materials = [
                            ("eye_hau", "はぅ", "Hau"),
                            ("eye_hachume", "はちゅ目", "Hachume"),
                            ("eye_nagomi", "なごみ", "Nagomi"),
                        ]

                    if append_materials:
                        # 目材質追加
                        for tex_name, mat_suffix, mat_suffix_english in append_materials:
                            shutil.copy(MFileUtils.resource_path(f"src/resources/{tex_name}.png"), tex_dir_path)
                            model.textures.append(os.path.join("tex", f"{tex_name}.png"))

                            add_material = copy.deepcopy(material)
                            add_material.name = f"{material.name}_{mat_suffix}"
                            add_material.english_name = f"{material_name}_{mat_suffix_english}"
                            add_material.texture_index = len(model.textures) - 1
                            add_material.alpha = 0
                            if (add_material.flag & 0x10) != 0:
                                add_material.flag -= 0x10  # エッジOFF
                            model.materials[add_material.name] = add_material
                            model.material_vertices[add_material.name] = []

                            for index_accessor, indices in indices_by_materials[material_name].items():
                                for v0_idx, v1_idx, v2_idx in zip(indices[:-2:3], indices[1:-1:3], indices[2::3]):
                                    # 面の貼り方がPMXは逆
                                    model.indices[index_idx] = [v2_idx, v1_idx, v0_idx]
                                    index_idx += 1

                                    if v0_idx not in model.material_vertices[add_material.name]:
                                        model.material_vertices[add_material.name].append(v0_idx)

                                    if v1_idx not in model.material_vertices[add_material.name]:
                                        model.material_vertices[add_material.name].append(v1_idx)

                                    if v2_idx not in model.material_vertices[add_material.name]:
                                        model.material_vertices[add_material.name].append(v2_idx)

                    if is_append_edge_material:

                        edge_material = copy.deepcopy(material)
                        edge_material.name = f"{material.name}_エッジ"
                        edge_material.english_name = f"{material_name}_Edge"
                        # エッジ色を拡散色と環境色に設定
                        edge_material.diffuse_color = MVector3D(material.edge_color.data()[:-1])
                        edge_material.ambient_color = edge_material.diffuse_color / 2
                        model.materials[edge_material.name] = edge_material
                        model.material_vertices[edge_material.name] = []

                        for index_accessor, indices in indices_by_materials[material_name].items():
                            for v0_idx, v1_idx, v2_idx in zip(indices[:-2:3], indices[1:-1:3], indices[2::3]):
                                # 頂点を複製
                                v0 = model.vertex_dict[v0_idx].copy()
                                v0.index = len(model.vertex_dict)
                                model.vertex_dict[v0.index] = v0

                                v1 = model.vertex_dict[v1_idx].copy()
                                v1.index = len(model.vertex_dict)
                                model.vertex_dict[v1.index] = v1

                                v2 = model.vertex_dict[v2_idx].copy()
                                v2.index = len(model.vertex_dict)
                                model.vertex_dict[v2.index] = v2

                                for v in [v0, v1, v2]:
                                    # 元の頂点の法線として保持
                                    original_normal = v.normal.copy()
                                    # 法線を反転させる
                                    v.normal *= -1
                                    v.normal.normalize()
                                    # 元の法線方向に少し拡大する
                                    v.position += original_normal * (material.edge_size * 0.02)

                                # 裏面として貼り付けるので、INDEXの順番はそのまま
                                model.indices[index_idx] = [v0.index, v1.index, v2.index]
                                index_idx += 1

                                if v0.index not in model.material_vertices[edge_material.name]:
                                    model.material_vertices[edge_material.name].append(v0.index)

                                if v1.index not in model.material_vertices[edge_material.name]:
                                    model.material_vertices[edge_material.name].append(v1.index)

                                if v2.index not in model.material_vertices[edge_material.name]:
                                    model.material_vertices[edge_material.name].append(v2.index)

        logger.info("-- 頂点・面・材質データ解析終了")

        return model

    def get_deform_index(
        self,
        vertex_idx: int,
        model: PmxModel,
        vertex_pos: MVector3D,
        joint: MVector4D,
        skin_joints: list,
        node_weight: list,
        bone_name_dict: dict,
    ):
        # まずは0じゃないデータ（何かしら有効なボーンINDEXがあるリスト）
        valiable_joints = np.where(joint.data() > 0)[0].tolist()
        # ウェイト
        org_weights = node_weight.data()[np.where(joint.data() > 0)]
        # ジョイント添え字からジョイントINDEXを取得(floatになってるのでint)
        org_joint_idxs = joint.data()[valiable_joints].astype(np.int)
        # 現行ボーンINDEXに置き換えたINDEX
        dest_joint_list = []
        for jidx in org_joint_idxs.tolist():
            for node_name, bone_param in bone_name_dict.items():
                if bone_param["node_index"] == skin_joints[jidx]:
                    dest_joint_list.append(model.bones[bone_param["name"]].index)
        dest_joints = np.array(dest_joint_list)

        # 腰は下半身に統合
        dest_joints = np.where(dest_joints == model.bones["腰"].index, model.bones["下半身"].index, dest_joints)

        # 下半身の上半身側は上半身に分散
        if model.bones["下半身"].index in dest_joints:
            trunk_distance = model.bones["上半身2"].position.y() - model.bones["上半身"].position.y()
            vector_trunk_distance = vertex_pos.y() - model.bones["上半身"].position.y()

            if np.sign(trunk_distance) == np.sign(vector_trunk_distance):
                # 上半身側の場合
                upper_trunk_factor = vector_trunk_distance / trunk_distance
                upper_trunk_weight_joints = np.where(dest_joints == model.bones["下半身"].index)[0]
                if len(upper_trunk_weight_joints) > 0:
                    if upper_trunk_factor > 1:
                        # 範囲より先の場合
                        dest_joints[upper_trunk_weight_joints] = model.bones["上半身2"].index
                    else:
                        # 下半身のウェイト値
                        dest_arm_weight = org_weights[upper_trunk_weight_joints]
                        # 上半身のウェイトは距離による
                        upper_weights = dest_arm_weight * upper_trunk_factor
                        # 下半身のウェイト値は残り
                        lower_weights = dest_arm_weight * (1 - upper_trunk_factor)

                        # FROMのウェイトを載せ替える
                        valiable_joints = valiable_joints + [model.bones["下半身"].index]
                        dest_joints[upper_trunk_weight_joints] = model.bones["下半身"].index
                        org_weights[upper_trunk_weight_joints] = lower_weights
                        # 腕捩のウェイトを追加する
                        valiable_joints = valiable_joints + [model.bones["上半身"].index]
                        dest_joints = np.append(dest_joints, model.bones["上半身"].index)
                        org_weights = np.append(org_weights, upper_weights)

        for direction in ["右", "左"]:
            # 足・ひざ・足首・つま先はそれぞれDに載せ替え
            for dest_bone_name, src_bone_name in [
                (f"{direction}足", f"{direction}足D"),
                (f"{direction}ひざ", f"{direction}ひざD"),
                (f"{direction}足首", f"{direction}足首D"),
                (f"{direction}つま先", f"{direction}足先EX"),
            ]:
                dest_joints = np.where(
                    dest_joints == model.bones[dest_bone_name].index, model.bones[src_bone_name].index, dest_joints
                )

            for base_from_name, base_to_name, base_twist_name in [("腕", "ひじ", "腕捩"), ("ひじ", "手首", "手捩")]:
                dest_arm_bone_name = f"{direction}{base_from_name}"
                dest_elbow_bone_name = f"{direction}{base_to_name}"
                dest_arm_twist1_bone_name = f"{direction}{base_twist_name}1"
                dest_arm_twist2_bone_name = f"{direction}{base_twist_name}2"
                dest_arm_twist3_bone_name = f"{direction}{base_twist_name}3"

                arm_elbow_distance = -1
                vector_arm_distance = 1

                # 腕捩に分散する
                if (
                    model.bones[dest_arm_bone_name].index in dest_joints
                    or model.bones[dest_arm_twist1_bone_name].index in dest_joints
                    or model.bones[dest_arm_twist2_bone_name].index in dest_joints
                    or model.bones[dest_arm_twist3_bone_name].index in dest_joints
                ):
                    # 腕に割り当てられているウェイトの場合
                    arm_elbow_distance = (
                        model.bones[dest_elbow_bone_name].position.x() - model.bones[dest_arm_bone_name].position.x()
                    )
                    vector_arm_distance = vertex_pos.x() - model.bones[dest_arm_bone_name].position.x()
                    twist_list = [
                        (dest_arm_twist1_bone_name, dest_arm_bone_name),
                        (dest_arm_twist2_bone_name, dest_arm_twist1_bone_name),
                        (dest_arm_twist3_bone_name, dest_arm_twist2_bone_name),
                    ]

                if np.sign(arm_elbow_distance) == np.sign(vector_arm_distance):
                    for dest_to_bone_name, dest_from_bone_name in twist_list:
                        # 腕からひじの間の頂点の場合
                        twist_distance = (
                            model.bones[dest_to_bone_name].position.x() - model.bones[dest_from_bone_name].position.x()
                        )
                        vector_distance = vertex_pos.x() - model.bones[dest_from_bone_name].position.x()
                        if np.sign(twist_distance) == np.sign(vector_distance):
                            # 腕から腕捩1の間にある頂点の場合
                            arm_twist_factor = vector_distance / twist_distance
                            # 腕が割り当てられているウェイトINDEX
                            arm_twist_weight_joints = np.where(dest_joints == model.bones[dest_from_bone_name].index)[
                                0
                            ]
                            if len(arm_twist_weight_joints) > 0:
                                if arm_twist_factor > 1:
                                    # 範囲より先の場合
                                    dest_joints[arm_twist_weight_joints] = model.bones[dest_to_bone_name].index
                                else:
                                    # 腕のウェイト値
                                    dest_arm_weight = org_weights[arm_twist_weight_joints]
                                    # 腕捩のウェイトはウェイト値の指定割合
                                    arm_twist_weights = dest_arm_weight * arm_twist_factor
                                    # 腕のウェイト値は残り
                                    arm_weights = dest_arm_weight * (1 - arm_twist_factor)

                                    # FROMのウェイトを載せ替える
                                    valiable_joints = valiable_joints + [model.bones[dest_from_bone_name].index]
                                    dest_joints[arm_twist_weight_joints] = model.bones[dest_from_bone_name].index
                                    org_weights[arm_twist_weight_joints] = arm_weights
                                    # 腕捩のウェイトを追加する
                                    valiable_joints = valiable_joints + [model.bones[dest_to_bone_name].index]
                                    dest_joints = np.append(dest_joints, model.bones[dest_to_bone_name].index)
                                    org_weights = np.append(org_weights, arm_twist_weights)

                                    logger.test(
                                        "[%s] from: %s, to: %s, factor: %s, dest_joints: %s, org_weights: %s",
                                        vertex_idx,
                                        dest_from_bone_name,
                                        dest_to_bone_name,
                                        arm_twist_factor,
                                        dest_joints,
                                        org_weights,
                                    )

        # 載せ替えた事で、ジョイントが重複している場合があるので、調整する
        joint_weights = {}
        for j, w in zip(dest_joints, org_weights):
            if j not in joint_weights:
                joint_weights[j] = 0
            joint_weights[j] += w

        # 対象となるウェイト値
        joint_values = list(joint_weights.keys())
        # 正規化(合計して1になるように)
        total_weights = np.array(list(joint_weights.values()))
        weight_values = (total_weights / total_weights.sum(axis=0, keepdims=1)).tolist()

        if len(joint_values) == 3:
            # 3つの場合、0を入れ込む
            return joint_values + [0], weight_values + [0]
        elif len(joint_values) > 4:
            # 4より多い場合、一番小さいのを捨てる（大体誤差）
            remove_idx = np.argmin(np.array(weight_values)).T
            del valiable_joints[remove_idx]
            del joint_values[remove_idx]
            del weight_values[remove_idx]

            # 正規化(合計して1になるように)
            total_weights = np.array(weight_values)
            weight_values = (total_weights / total_weights.sum(axis=0, keepdims=1)).tolist()

        return joint_values, weight_values

    def convert_bone(self, model: PmxModel):
        if "nodes" not in model.json_data:
            logger.error("変換可能なボーン情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
            return None, None

        # 表示枠 ------------------------
        model.display_slots["全ての親"] = DisplaySlot("Root", "Root", 1, 1)
        model.display_slots["全ての親"].references.append((0, 0))

        # モーフの表示枠
        model.display_slots["表情"] = DisplaySlot("表情", "Exp", 1, 1)

        node_dict = {}
        node_name_dict = {}
        for nidx, node in enumerate(model.json_data["nodes"]):
            node_translation = MVector3D() if "translation" not in node else MVector3D(*node["translation"])

            node = model.json_data["nodes"][nidx]
            logger.debug(f"[{nidx:03d}] node: {node}")

            node_name = node["name"]

            # 位置
            position = node_translation * MIKU_METER * MVector3D(-1, 1, 1)

            children = node["children"] if "children" in node else []

            node_dict[nidx] = {
                "name": node_name,
                "relative_position": position,
                "position": position,
                "parent": -1,
                "children": children,
            }
            node_name_dict[node_name] = nidx

        # 親子関係設定
        for nidx, node_param in node_dict.items():
            for midx, parent_node_param in node_dict.items():
                if nidx in parent_node_param["children"]:
                    node_dict[nidx]["parent"] = midx

        # 絶対位置計算
        for nidx, node_param in node_dict.items():
            node_dict[nidx]["position"] = self.calc_bone_position(model, node_dict, node_param)

        # まずは人体ボーン
        bone_name_dict = {}
        for node_name, bone_param in BONE_PAIRS.items():
            parent_name = BONE_PAIRS[bone_param["parent"]]["name"] if bone_param["parent"] else None
            parent_index = model.bones[parent_name].index if parent_name else -1

            node_index = -1
            position = MVector3D()
            bone = Bone(bone_param["name"], node_name, position, parent_index, 0, bone_param["flag"])
            if bone.name in model.bones:
                # 同一名が既にある場合、スルー（尻対策）
                continue

            if parent_index >= 0:
                if node_name in node_name_dict:
                    node_index = node_name_dict[node_name]
                    position = node_dict[node_name_dict[node_name]]["position"].copy()
                elif node_name == "Center":
                    position = node_dict[node_name_dict["J_Bip_C_Hips"]]["position"] * 0.7
                elif node_name == "Groove":
                    position = node_dict[node_name_dict["J_Bip_C_Hips"]]["position"] * 0.8
                elif node_name == "J_Bip_C_Spine2":
                    position = node_dict[node_name_dict["J_Bip_C_Spine"]]["position"].copy()
                elif node_name in ["J_Adj_FaceEyeHighlight", "J_Adj_FaceEye"]:
                    position = node_dict[node_name_dict["J_Adj_L_FaceEye"]]["position"] + (
                        (
                            node_dict[node_name_dict["J_Adj_R_FaceEye"]]["position"]
                            - node_dict[node_name_dict["J_Adj_L_FaceEye"]]["position"]
                        )
                        * 0.5
                    )
                elif "shoulderP_" in node_name:
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_Shoulder"]]["position"].copy()
                elif "shoulderC_" in node_name:
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_UpperArm"]]["position"].copy()
                    bone.effect_index = bone_name_dict[f"shoulderP_{node_name[-1]}"]["index"]
                    bone.effect_factor = -1
                elif "arm_twist_" in node_name:
                    factor = 0.25 if node_name[-2] == "1" else 0.75 if node_name[-2] == "3" else 0.5
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_UpperArm"]]["position"] + (
                        (
                            node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_LowerArm"]]["position"]
                            - node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_UpperArm"]]["position"]
                        )
                        * factor
                    )
                    if node_name[-2] in ["1", "2", "3"]:
                        bone.effect_index = bone_name_dict[f"arm_twist_{node_name[-1]}"]["index"]
                        bone.effect_factor = factor
                elif "wrist_twist_" in node_name:
                    factor = 0.25 if node_name[-2] == "1" else 0.75 if node_name[-2] == "3" else 0.5
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_LowerArm"]]["position"] + (
                        (
                            node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_Hand"]]["position"]
                            - node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_LowerArm"]]["position"]
                        )
                        * factor
                    )
                    if node_name[-2] in ["1", "2", "3"]:
                        bone.effect_index = bone_name_dict[f"wrist_twist_{node_name[-1]}"]["index"]
                        bone.effect_factor = factor
                elif "waistCancel_" in node_name:
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_UpperLeg"]]["position"].copy()
                    bone.effect_index = bone_name_dict["J_Bip_C_Hips"]["index"]
                    bone.effect_factor = -1
                elif "leg_IK_Parent_" in node_name:
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_Foot"]]["position"].copy()
                    position.setY(0)
                elif "leg_IK_" in node_name:
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_Foot"]]["position"].copy()
                elif "toe_IK_" in node_name:
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_ToeBase"]]["position"].copy()
                elif "leg_D_" in node_name:
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_UpperLeg"]]["position"].copy()
                    bone.effect_index = bone_name_dict[f"J_Bip_{node_name[-1]}_UpperLeg"]["index"]
                    bone.effect_factor = 1
                    bone.layer = 1
                elif "knee_D_" in node_name:
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_LowerLeg"]]["position"].copy()
                    bone.effect_index = bone_name_dict[f"J_Bip_{node_name[-1]}_LowerLeg"]["index"]
                    bone.effect_factor = 1
                    bone.layer = 1
                elif "ankle_D_" in node_name:
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_Foot"]]["position"].copy()
                    bone.effect_index = bone_name_dict[f"J_Bip_{node_name[-1]}_Foot"]["index"]
                    bone.effect_factor = 1
                    bone.layer = 1
                elif "toe_EX_" in node_name:
                    position = node_dict[node_name_dict[f"J_Bip_{node_name[-1]}_ToeBase"]]["position"].copy()
                    bone.layer = 1
            bone.position = position
            bone.index = len(model.bones)

            # 表示枠
            if bone_param["display"]:
                if bone_param["display"] not in model.display_slots:
                    model.display_slots[bone_param["display"]] = DisplaySlot(
                        bone_param["display"], bone_param["display"], 0, 0
                    )
                model.display_slots[bone_param["display"]].references.append((0, bone.index))

            model.bones[bone.name] = bone
            bone_name_dict[node_name] = {
                "index": bone.index,
                "name": bone.name,
                "node_name": node_name,
                "node_index": node_index,
            }

        if "髪" not in model.display_slots:
            model.display_slots["髪"] = DisplaySlot("髪", "Hair", 0, 0)
        if "その他" not in model.display_slots:
            model.display_slots["その他"] = DisplaySlot("その他", "Other", 0, 0)

        # 人体以外のボーン
        hair_blocks = []
        other_blocks = []
        for nidx, node_param in node_dict.items():
            if node_param["name"] not in bone_name_dict:
                bone = Bone(node_param["name"], node_param["name"], node_param["position"], -1, 0, 0x0002)
                parent_index = (
                    bone_name_dict[node_dict[node_param["parent"]]["name"]]["index"]
                    if node_param["parent"] in node_dict and node_dict[node_param["parent"]]["name"] in bone_name_dict
                    else -1
                )
                bone.parent_index = parent_index
                bone.index = len(model.bones)

                if node_param["name"] not in DISABLE_BONES:
                    # 1.4.1でボーン名が短くなったのでそれに合わせて調整
                    node_names = (
                        node_param["name"].split("_")
                        if "Hair" in node_param["name"]
                        else node_param["name"].split("_Sec_")
                        if "_Sec_" in node_param["name"]
                        else node_param["name"].split("J_")
                    )
                    bone_block = None
                    bone_name = None

                    if "Hair" in node_param["name"]:
                        if len(hair_blocks) == 0:
                            bone_block = {"bone_block_name": "髪", "bone_block_size": 1, "size": 1}
                        else:
                            bone_block = {
                                "bone_block_name": "髪",
                                "bone_block_size": hair_blocks[-1]["bone_block_size"],
                                "size": hair_blocks[-1]["size"] + 1,
                            }
                        hair_blocks.append(bone_block)
                    else:
                        if len(other_blocks) == 0:
                            bone_block = {"bone_block_name": "装飾", "bone_block_size": 1, "size": 1}
                        else:
                            bone_block = {
                                "bone_block_name": "装飾",
                                "bone_block_size": other_blocks[-1]["bone_block_size"],
                                "size": other_blocks[-1]["size"] + 1,
                            }
                        other_blocks.append(bone_block)
                    bone_name = (
                        f'{bone_block["bone_block_name"]}_{bone_block["bone_block_size"]:02d}-{bone_block["size"]:02d}'
                    )

                    if "Hair" not in node_param["name"] and len(node_names) > 1:
                        # 装飾の場合、末尾を入れる
                        bone_name += node_param["name"][len(node_names[0]) :]

                    bone.name = bone_name

                model.bones[bone.name] = bone
                bone_name_dict[node_param["name"]] = {
                    "index": bone.index,
                    "name": bone.name,
                    "node_name": node_param["name"],
                    "node_index": node_name_dict[node_param["name"]],
                }

                if node_param["name"] not in DISABLE_BONES:
                    if len(node_param["children"]) == 0:
                        # 末端の場合次ボーンで段を変える(加算用にsizeは0)
                        if "Hair" in node_param["name"]:
                            hair_blocks.append(
                                {
                                    "bone_block_name": "髪",
                                    "bone_block_size": hair_blocks[-1]["bone_block_size"] + 1,
                                    "size": 0,
                                }
                            )
                        else:
                            other_blocks.append(
                                {
                                    "bone_block_name": "装飾",
                                    "bone_block_size": other_blocks[-1]["bone_block_size"] + 1,
                                    "size": 0,
                                }
                            )

        # ローカル軸・IK設定
        for bone in model.bones.values():
            model.bone_indexes[bone.index] = bone.name

            # 人体ボーン
            if bone.english_name in BONE_PAIRS:
                # 表示先
                tail = BONE_PAIRS[bone.english_name]["tail"]
                if tail:
                    if type(tail) is MVector3D:
                        bone.tail_position = tail.copy()
                    else:
                        bone.tail_index = bone_name_dict[tail]["index"]
                if bone.name == "下半身":
                    # 腰は表示順が上なので、相対指定
                    bone.tail_position = model.bones["腰"].position - bone.position
                elif bone.name == "頭":
                    bone.tail_position = MVector3D(0, 1, 0)

                direction = bone.name[0]

                # 足IK
                leg_name = f"{direction}足"
                knee_name = f"{direction}ひざ"
                ankle_name = f"{direction}足首"
                toe_name = f"{direction}つま先"

                if (
                    bone.name in ["右足ＩＫ", "左足ＩＫ"]
                    and leg_name in model.bones
                    and knee_name in model.bones
                    and ankle_name in model.bones
                ):
                    leg_ik_link = []
                    leg_ik_link.append(
                        IkLink(
                            model.bones[knee_name].index,
                            1,
                            MVector3D(math.radians(-180), 0, 0),
                            MVector3D(math.radians(-0.5), 0, 0),
                        )
                    )
                    leg_ik_link.append(IkLink(model.bones[leg_name].index, 0))
                    leg_ik = Ik(model.bones[ankle_name].index, 40, 1, leg_ik_link)
                    bone.ik = leg_ik

                if bone.name in ["右つま先ＩＫ", "左つま先ＩＫ"] and ankle_name in model.bones and toe_name in model.bones:
                    toe_ik_link = []
                    toe_ik_link.append(IkLink(model.bones[ankle_name].index, 0))
                    toe_ik = Ik(model.bones[toe_name].index, 40, 1, toe_ik_link)
                    bone.ik = toe_ik

                if bone.name in ["上半身3"] and "上半身2" in model.bones:
                    bone.flag |= 0x0100
                    bone.effect_index = model.bones["上半身2"].index
                    bone.effect_factor = 0.4

                if bone.name in ["右目", "左目"] and "両目" in model.bones:
                    bone.flag |= 0x0100
                    bone.effect_index = model.bones["両目"].index
                    bone.effect_factor = 0.3

                if bone.name in ["両目光"] and "両目" in model.bones:
                    bone.flag |= 0x0100
                    bone.effect_index = model.bones["両目"].index
                    bone.effect_factor = 1

                if bone.name in ["右目光", "左目光"] and "両目光" in model.bones:
                    bone.flag |= 0x0100
                    bone.effect_index = model.bones["両目光"].index
                    bone.effect_factor = 0.3
            else:
                # 人体以外
                # 表示先
                node_param = node_dict[node_name_dict[bone.english_name]]
                tail_index = (
                    bone_name_dict[node_dict[node_param["children"][0]]["name"]]["index"]
                    if node_param["children"]
                    and node_param["children"][0] in node_dict
                    and node_dict[node_param["children"][0]]["name"] in bone_name_dict
                    else -1
                )
                if tail_index >= 0:
                    bone.tail_index = tail_index
                    bone.flag |= 0x0001 | 0x0008 | 0x0010
                elif "Glasses" in bone.name:
                    # メガネは単体で操作可能(+移動)
                    bone.flag |= 0x0004 | 0x0008 | 0x0010

                if bone.getVisibleFlag():
                    if "Hair" in bone.english_name:
                        model.display_slots["髪"].references.append((0, bone.index))
                    else:
                        model.display_slots["その他"].references.append((0, bone.index))

        logger.info("-- ボーンデータ解析終了")

        return model, bone_name_dict

    def calc_bone_position(self, model: PmxModel, node_dict: dict, node_param: dict):
        if node_param["parent"] == -1:
            return node_param["relative_position"]

        return node_param["relative_position"] + self.calc_bone_position(
            model, node_dict, node_dict[node_param["parent"]]
        )

    def create_model(self):
        model = PmxModel()

        # テクスチャ用ディレクトリ
        tex_dir_path = os.path.join(str(Path(self.options.output_path).resolve().parents[0]), "tex")
        os.makedirs(tex_dir_path, exist_ok=True)
        # 展開用ディレクトリ作成
        glft_dir_path = os.path.join(str(Path(self.options.output_path).resolve().parents[0]), "glTF")
        os.makedirs(glft_dir_path, exist_ok=True)
        # PmxTailor設定用ディレクトリ作成
        setting_dir_path = os.path.join(str(Path(self.options.output_path).resolve().parents[0]), "PmxTailorSetting")
        os.makedirs(setting_dir_path, exist_ok=True)

        with open(self.options.vrm_model.path, "rb") as f:
            self.buffer = f.read()

            signature = self.unpack(12, "12s")
            logger.test("signature: %s (%s)", signature, self.offset)

            # JSON文字列読み込み
            json_buf_size = self.unpack(8, "L")
            json_text = self.read_text(json_buf_size)

            model.json_data = json.loads(json_text)

            # JSON出力
            jf = open(os.path.join(glft_dir_path, "gltf.json"), "w", encoding="utf-8")
            json.dump(model.json_data, jf, ensure_ascii=False, indent=4, sort_keys=True, separators=(",", ": "))
            logger.info("-- JSON出力終了")

            if (
                "extensions" not in model.json_data
                or "VRM" not in model.json_data["extensions"]
                or "exporterVersion" not in model.json_data["extensions"]["VRM"]
            ):
                logger.error("出力ソフト情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
                return None, None, None

            if (
                "extensions" not in model.json_data
                or "VRM" not in model.json_data["extensions"]
                or "meta" not in model.json_data["extensions"]["VRM"]
            ):
                logger.error("メタ情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
                return None, None, None

            if "VRoidStudio-0." in model.json_data["extensions"]["VRM"]["exporterVersion"]:
                # VRoid Studioベータ版はNG
                logger.error(
                    "VRoid Studio ベータ版 で出力されたvrmデータではあるため、処理を中断します。\n正式版でコンバートしてから再度試してください。\n出力元: %s",
                    model.json_data["extensions"]["VRM"]["exporterVersion"],
                    decoration=MLogger.DECORATION_BOX,
                )
                return None, None, None

            if "VRoid Studio-1." not in model.json_data["extensions"]["VRM"]["exporterVersion"]:
                # VRoid Studio正式版じゃなくても警告だけに留める
                logger.warning(
                    "VRoid Studio 1.x で出力されたvrmデータではないため、結果がおかしくなる可能性があります。\n（結果がおかしくてもサポート対象外となります）\n出力元: %s",
                    model.json_data["extensions"]["VRM"]["exporterVersion"],
                    decoration=MLogger.DECORATION_BOX,
                )

            if "title" in model.json_data["extensions"]["VRM"]["meta"]:
                model.name = model.json_data["extensions"]["VRM"]["meta"]["title"]
                model.english_name = model.json_data["extensions"]["VRM"]["meta"]["title"]
            if not model.name:
                # titleにモデル名が入ってなかった場合、ファイル名を代理入力
                file_name = os.path.basename(self.options.vrm_model.path).split(".")[0]
                model.name = file_name
                model.english_name = file_name

            model.comment += f"{logger.transtext('PMX出力')}: Vroid2Pmx {self.options.version_name}\r\n"

            model.comment += f"\r\n{logger.transtext('アバター情報')} -------\r\n"

            if "author" in model.json_data["extensions"]["VRM"]["meta"]:
                model.comment += (
                    f"{logger.transtext('作者')}: {model.json_data['extensions']['VRM']['meta']['author']}\r\n"
                )
            if "contactInformation" in model.json_data["extensions"]["VRM"]["meta"]:
                model.comment += f"{logger.transtext('連絡先')}: {model.json_data['extensions']['VRM']['meta']['contactInformation']}\r\n"
            if "reference" in model.json_data["extensions"]["VRM"]["meta"]:
                model.comment += (
                    f"{logger.transtext('参照')}: {model.json_data['extensions']['VRM']['meta']['reference']}\r\n"
                )
            if "version" in model.json_data["extensions"]["VRM"]["meta"]:
                model.comment += (
                    f"{logger.transtext('バージョン')}: {model.json_data['extensions']['VRM']['meta']['version']}\r\n"
                )

            model.comment += f"\r\n{logger.transtext('アバターの人格に関する許諾範囲')} -------\r\n"

            if "allowedUserName" in model.json_data["extensions"]["VRM"]["meta"]:
                model.comment += f"{logger.transtext('アバターに人格を与えることの許諾範囲')}: {model.json_data['extensions']['VRM']['meta']['allowedUserName']}\r\n"
            if "violentUssageName" in model.json_data["extensions"]["VRM"]["meta"]:
                model.comment += f"{logger.transtext('このアバターを用いて暴力表現を演じることの許可')}: {model.json_data['extensions']['VRM']['meta']['violentUssageName']}\r\n"
            if "sexualUssageName" in model.json_data["extensions"]["VRM"]["meta"]:
                model.comment += f"{logger.transtext('このアバターを用いて性的表現を演じることの許可')}: {model.json_data['extensions']['VRM']['meta']['sexualUssageName']}\r\n"
            if "commercialUssageName" in model.json_data["extensions"]["VRM"]["meta"]:
                model.comment += f"{logger.transtext('商用利用の許可')}: {model.json_data['extensions']['VRM']['meta']['commercialUssageName']}\r\n"
            if "otherPermissionUrl" in model.json_data["extensions"]["VRM"]["meta"]:
                model.comment += f"{logger.transtext('その他のライセンス条件')}: {model.json_data['extensions']['VRM']['meta']['otherPermissionUrl']}\r\n"

            model.comment += f"\r\n{logger.transtext('再配布・改変に関する許諾範囲')} -------\r\n"

            if "licenseName" in model.json_data["extensions"]["VRM"]["meta"]:
                model.comment += f"{logger.transtext('ライセンスタイプ')}: {model.json_data['extensions']['VRM']['meta']['licenseName']}\r\n"
            if "otherPermissionUrl" in model.json_data["extensions"]["VRM"]["meta"]:
                model.comment += f"{logger.transtext('その他のライセンス条件')}: {model.json_data['extensions']['VRM']['meta']['otherPermissionUrl']}\r\n"

            # binデータ
            bin_buf_size = self.unpack(8, "L")
            logger.debug(f"bin_buf_size: {bin_buf_size}")

            with open(os.path.join(glft_dir_path, "data.bin"), "wb") as bf:
                bf.write(self.buffer[self.offset : (self.offset + bin_buf_size)])

            # 空値をスフィア用に登録
            model.textures.append("")

            if "images" not in model.json_data:
                logger.error("変換可能な画像情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
                return None, None, None

            # jsonデータの中に画像データの指定がある場合
            image_offset = 0
            for image in model.json_data["images"]:
                if int(image["bufferView"]) < len(model.json_data["bufferViews"]):
                    image_buffer = model.json_data["bufferViews"][int(image["bufferView"])]
                    # 画像の開始位置はオフセット分ずらす
                    image_start = self.offset + image_buffer["byteOffset"]
                    # 拡張子
                    ext = MIME_TYPE[image["mimeType"]]
                    # 画像名
                    image_name = f"{image['name']}.{ext}"
                    with open(os.path.join(glft_dir_path, image_name), "wb") as ibf:
                        ibf.write(self.buffer[image_start : (image_start + image_buffer["byteLength"])])
                    # オフセット加算
                    image_offset += image_buffer["byteLength"]
                    # PMXに追記
                    model.textures.append(os.path.join("tex", image_name))
                    # テクスチャコピー
                    shutil.copy(os.path.join(glft_dir_path, image_name), os.path.join(tex_dir_path, image_name))

            logger.info("-- テクスチャデータ解析終了")

        return model, tex_dir_path, setting_dir_path

    # アクセサ経由で値を取得する
    # https://github.com/ft-lab/Documents_glTF/blob/master/structure.md
    def read_from_accessor(self, model: PmxModel, accessor_idx: int):
        bresult = None
        aidx = 0
        if accessor_idx < len(model.json_data["accessors"]):
            accessor = model.json_data["accessors"][accessor_idx]
            acc_type = accessor["type"]
            if accessor["bufferView"] < len(model.json_data["bufferViews"]):
                buffer = model.json_data["bufferViews"][accessor["bufferView"]]
                logger.debug("accessor: %s, %s", accessor_idx, buffer)
                if "count" in accessor:
                    bresult = []
                    if acc_type == "VEC3":
                        buf_type, buf_num = self.define_buf_type(accessor["componentType"])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor["count"]):
                            buf_start = self.offset + buffer["byteOffset"] + ((buf_num * 3) * n)

                            # Vec3 / float
                            xresult = struct.unpack_from(buf_type, self.buffer, buf_start)
                            yresult = struct.unpack_from(buf_type, self.buffer, buf_start + buf_num)
                            zresult = struct.unpack_from(buf_type, self.buffer, buf_start + (buf_num * 2))

                            if buf_type == "f":
                                bresult.append(MVector3D(float(xresult[0]), float(yresult[0]), float(zresult[0])))
                            else:
                                bresult.append(MVector3D(int(xresult[0]), int(yresult[0]), int(zresult[0])))

                            aidx += 1

                            if aidx % 5000 == 0:
                                logger.info("-- -- Accessor[%s/%s/%s][%s]", accessor_idx, acc_type, buf_type, aidx)
                            else:
                                logger.test("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                    elif acc_type == "VEC2":
                        buf_type, buf_num = self.define_buf_type(accessor["componentType"])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor["count"]):
                            buf_start = self.offset + buffer["byteOffset"] + ((buf_num * 2) * n)

                            # Vec3 / float
                            xresult = struct.unpack_from(buf_type, self.buffer, buf_start)
                            yresult = struct.unpack_from(buf_type, self.buffer, buf_start + buf_num)

                            bresult.append(MVector2D(float(xresult[0]), float(yresult[0])))

                            aidx += 1

                            if aidx % 5000 == 0:
                                logger.info("-- -- Accessor[%s/%s/%s][%s]", accessor_idx, acc_type, buf_type, aidx)
                            else:
                                logger.test("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                    elif acc_type == "VEC4":
                        buf_type, buf_num = self.define_buf_type(accessor["componentType"])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor["count"]):
                            buf_start = self.offset + buffer["byteOffset"] + ((buf_num * 4) * n)

                            # Vec3 / float
                            xresult = struct.unpack_from(buf_type, self.buffer, buf_start)
                            yresult = struct.unpack_from(buf_type, self.buffer, buf_start + buf_num)
                            zresult = struct.unpack_from(buf_type, self.buffer, buf_start + (buf_num * 2))
                            wresult = struct.unpack_from(buf_type, self.buffer, buf_start + (buf_num * 3))

                            if buf_type == "f":
                                bresult.append(
                                    MVector4D(
                                        float(xresult[0]), float(yresult[0]), float(zresult[0]), float(wresult[0])
                                    )
                                )
                            else:
                                bresult.append(
                                    MVector4D(int(xresult[0]), int(yresult[0]), int(zresult[0]), int(wresult[0]))
                                )

                            aidx += 1

                            if aidx % 5000 == 0:
                                logger.info("-- -- Accessor[%s/%s/%s][%s]", accessor_idx, acc_type, buf_type, aidx)
                            else:
                                logger.test("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                    elif acc_type == "SCALAR":
                        buf_type, buf_num = self.define_buf_type(accessor["componentType"])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor["count"]):
                            buf_start = self.offset + buffer["byteOffset"] + (buf_num * n)
                            xresult = struct.unpack_from(buf_type, self.buffer, buf_start)

                            if buf_type == "f":
                                bresult.append(float(xresult[0]))
                            else:
                                bresult.append(int(xresult[0]))

                            aidx += 1

                            if aidx % 5000 == 0:
                                logger.info("-- -- Accessor[%s/%s/%s][%s]", accessor_idx, acc_type, buf_type, aidx)
                            else:
                                logger.test("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

        return bresult

    def define_buf_type(self, componentType: int):
        if componentType == 5120:
            return "b", 1
        elif componentType == 5121:
            return "B", 1
        elif componentType == 5122:
            return "h", 2
        elif componentType == 5123:
            return "H", 2
        elif componentType == 5124:
            return "i", 4
        elif componentType == 5125:
            return "I", 4

        return "f", 4

    def read_text(self, format_size):
        bresult = self.unpack(format_size, "{0}s".format(format_size))
        return bresult.decode("UTF8")

    # 解凍して、offsetを更新する
    def unpack(self, format_size, format):
        bresult = struct.unpack_from(format, self.buffer, self.offset)

        # オフセットを更新する
        self.offset += format_size

        if bresult:
            result = bresult[0]
        else:
            result = None

        return result


def calc_ratio(ratio: float, oldmin: float, oldmax: float, newmin: float, newmax: float):
    # https://qastack.jp/programming/929103/convert-a-number-range-to-another-range-maintaining-ratio
    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return (((ratio - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin


def randomname(n) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


# https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
def calc_intersect_point(p0: np.ndarray, p1: np.ndarray, p_co: np.ndarray, p_no: np.ndarray, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = p1 - p0
    dot = p_no.dot(u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = p0 - p_co
        fac = -p_no.dot(w) / dot
        u = u * fac
        return p0 + u

    return None


# https://stackoverflow.com/questions/18838403/get-the-closest-point-to-a-plane-defined-by-four-vertices-in-python
def calc_nearest_point(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p: np.ndarray):
    u = p1 - p0
    v = p2 - p0
    # vector normal to plane
    n = np.cross(u, v)
    n /= np.linalg.norm(n)

    p_ = p - p0
    # dist_to_plane = np.dot(p_, n)
    p_normal = np.dot(p_, n) * n
    p_tangent = p_ - p_normal

    closest_point = p_tangent + p0
    # coords = np.linalg.lstsq(np.column_stack((u, v)), p_tangent)[0]

    return closest_point


# https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
def polyfit2d(x, y, z, order=3):
    ncols = (order + 1) ** 2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m


def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))
    z = np.zeros_like(x)
    for a, (i, j) in zip(m, ij):
        z += a * x**i * y**j
    return z


DISABLE_BONES = [
    "Face",
    "Body",
    "Hair",
    "Hairs",
    "Hair001",
    "secondary",
]

BONE_PAIRS = {
    "Root": {
        "name": "全ての親",
        "parent": None,
        "tail": "Center",
        "display": None,
        "flag": 0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0010,
    },
    "Center": {
        "name": "センター",
        "parent": "Root",
        "tail": None,
        "display": "センター",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010,
    },
    "Groove": {
        "name": "グルーブ",
        "parent": "Center",
        "tail": None,
        "display": "センター",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010,
    },
    "J_Bip_C_Hips": {
        "name": "腰",
        "parent": "Groove",
        "tail": None,
        "display": "体幹",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010,
    },
    "J_Bip_C_Spine": {
        "name": "下半身",
        "parent": "J_Bip_C_Hips",
        "tail": None,
        "display": "体幹",
        "flag": 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_C_Spine2": {
        "name": "上半身",
        "parent": "J_Bip_C_Hips",
        "tail": "J_Bip_C_Chest",
        "display": "体幹",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_C_Chest": {
        "name": "上半身2",
        "parent": "J_Bip_C_Spine2",
        "tail": "J_Bip_C_UpperChest",
        "display": "体幹",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_C_UpperChest": {
        "name": "上半身3",
        "parent": "J_Bip_C_Chest",
        "tail": "J_Bip_C_Neck",
        "display": "体幹",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_C_Neck": {
        "name": "首",
        "parent": "J_Bip_C_UpperChest",
        "tail": "J_Bip_C_Head",
        "display": "体幹",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_C_Head": {
        "name": "頭",
        "parent": "J_Bip_C_Neck",
        "tail": None,
        "display": "体幹",
        "flag": 0x0002 | 0x0008 | 0x0010,
    },
    "J_Adj_FaceEye": {
        "name": "両目",
        "parent": "J_Bip_C_Head",
        "tail": None,
        "display": "顔",
        "flag": 0x0002 | 0x0008 | 0x0010,
    },
    "J_Adj_L_FaceEye": {
        "name": "左目",
        "parent": "J_Bip_C_Head",
        "tail": None,
        "display": "顔",
        "flag": 0x0002 | 0x0008 | 0x0010,
    },
    "J_Adj_R_FaceEye": {
        "name": "右目",
        "parent": "J_Bip_C_Head",
        "tail": None,
        "display": "顔",
        "flag": 0x0002 | 0x0008 | 0x0010,
    },
    "J_Adj_FaceEyeHighlight": {
        "name": "両目光",
        "parent": "J_Adj_FaceEye",
        "tail": None,
        "display": "顔",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010,
    },
    "J_Adj_L_FaceEyeHighlight": {
        "name": "左目光",
        "parent": "J_Bip_C_Head",
        "tail": None,
        "display": "顔",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010,
    },
    "J_Adj_R_FaceEyeHighlight": {
        "name": "右目光",
        "parent": "J_Bip_C_Head",
        "tail": None,
        "display": "顔",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010,
    },
    "J_Adj_FaceTongue1": {
        "name": "舌1",
        "parent": "J_Bip_C_Head",
        "tail": "J_Adj_FaceTongue2",
        "display": "顔",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Adj_FaceTongue2": {
        "name": "舌2",
        "parent": "J_Adj_FaceTongue1",
        "tail": "J_Adj_FaceTongue3",
        "display": "顔",
        "flag": 0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Adj_FaceTongue3": {
        "name": "舌3",
        "parent": "J_Adj_FaceTongue2",
        "tail": "J_Adj_FaceTongue4",
        "display": "顔",
        "flag": 0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Adj_FaceTongue4": {
        "name": "舌4",
        "parent": "J_Adj_FaceTongue3",
        "tail": None,
        "display": "顔",
        "flag": 0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Sec_L_Bust1": {
        "name": "左胸",
        "parent": "J_Bip_C_UpperChest",
        "tail": "J_Sec_L_Bust2",
        "display": "胸",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Sec_L_Bust2": {
        "name": "左胸先",
        "parent": "J_Sec_L_Bust1",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "J_Sec_R_Bust1": {
        "name": "右胸",
        "parent": "J_Bip_C_UpperChest",
        "tail": "J_Sec_R_Bust2",
        "display": "胸",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Sec_R_Bust2": {
        "name": "右胸先",
        "parent": "J_Sec_R_Bust1",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "shoulderP_L": {
        "name": "左肩P",
        "parent": "J_Bip_C_UpperChest",
        "tail": None,
        "display": "左手",
        "flag": 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_L_Shoulder": {
        "name": "左肩",
        "parent": "shoulderP_L",
        "tail": "J_Bip_L_UpperArm",
        "display": "左手",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "shoulderC_L": {
        "name": "左肩C",
        "parent": "J_Bip_L_Shoulder",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "J_Bip_L_UpperArm": {
        "name": "左腕",
        "parent": "shoulderC_L",
        "tail": "J_Bip_L_LowerArm",
        "display": "左手",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "arm_twist_L": {
        "name": "左腕捩",
        "parent": "J_Bip_L_UpperArm",
        "tail": None,
        "display": "左手",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800 | 0x0800,
    },
    "arm_twist_1L": {
        "name": "左腕捩1",
        "parent": "J_Bip_L_UpperArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "arm_twist_2L": {
        "name": "左腕捩2",
        "parent": "J_Bip_L_UpperArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "arm_twist_3L": {
        "name": "左腕捩3",
        "parent": "J_Bip_L_UpperArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "J_Bip_L_LowerArm": {
        "name": "左ひじ",
        "parent": "arm_twist_L",
        "tail": "J_Bip_L_Hand",
        "display": "左手",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "wrist_twist_L": {
        "name": "左手捩",
        "parent": "J_Bip_L_LowerArm",
        "tail": None,
        "display": "左手",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800,
    },
    "wrist_twist_1L": {
        "name": "左手捩1",
        "parent": "J_Bip_L_LowerArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "wrist_twist_2L": {
        "name": "左手捩2",
        "parent": "J_Bip_L_LowerArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "wrist_twist_3L": {
        "name": "左手捩3",
        "parent": "J_Bip_L_LowerArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "J_Bip_L_Hand": {
        "name": "左手首",
        "parent": "wrist_twist_L",
        "tail": None,
        "display": "左手",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Thumb1": {
        "name": "左親指０",
        "parent": "J_Bip_L_Hand",
        "tail": "J_Bip_L_Thumb2",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Thumb2": {
        "name": "左親指１",
        "parent": "J_Bip_L_Thumb1",
        "tail": "J_Bip_L_Thumb3",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Thumb3": {
        "name": "左親指２",
        "parent": "J_Bip_L_Thumb2",
        "tail": "J_Bip_L_Thumb3_end",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Thumb3_end": {
        "name": "左親指先",
        "parent": "J_Bip_L_Thumb3",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "J_Bip_L_Index1": {
        "name": "左人指１",
        "parent": "J_Bip_L_Hand",
        "tail": "J_Bip_L_Index2",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Index2": {
        "name": "左人指２",
        "parent": "J_Bip_L_Index1",
        "tail": "J_Bip_L_Index3",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Index3": {
        "name": "左人指３",
        "parent": "J_Bip_L_Index2",
        "tail": "J_Bip_L_Index3_end",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Index3_end": {
        "name": "左人指先",
        "parent": "J_Bip_L_Index3",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "J_Bip_L_Middle1": {
        "name": "左中指１",
        "parent": "J_Bip_L_Hand",
        "tail": "J_Bip_L_Middle2",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Middle2": {
        "name": "左中指２",
        "parent": "J_Bip_L_Middle1",
        "tail": "J_Bip_L_Middle3",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Middle3": {
        "name": "左中指３",
        "parent": "J_Bip_L_Middle2",
        "tail": "J_Bip_L_Middle3_end",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Middle3_end": {
        "name": "左中指先",
        "parent": "J_Bip_L_Middle3",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "J_Bip_L_Ring1": {
        "name": "左薬指１",
        "parent": "J_Bip_L_Hand",
        "tail": "J_Bip_L_Ring2",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Ring2": {
        "name": "左薬指２",
        "parent": "J_Bip_L_Ring1",
        "tail": "J_Bip_L_Ring3",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Ring3": {
        "name": "左薬指３",
        "parent": "J_Bip_L_Ring2",
        "tail": "J_Bip_L_Ring3_end",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Ring3_end": {
        "name": "左薬指先",
        "parent": "J_Bip_L_Ring3",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "J_Bip_L_Little1": {
        "name": "左小指１",
        "parent": "J_Bip_L_Hand",
        "tail": "J_Bip_L_Little2",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Little2": {
        "name": "左小指２",
        "parent": "J_Bip_L_Little1",
        "tail": "J_Bip_L_Little3",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Little3": {
        "name": "左小指３",
        "parent": "J_Bip_L_Little2",
        "tail": "J_Bip_L_Little3_end",
        "display": "左指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_L_Little3_end": {
        "name": "左小指先",
        "parent": "J_Bip_L_Little3",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "shoulderP_R": {
        "name": "右肩P",
        "parent": "J_Bip_C_UpperChest",
        "tail": None,
        "display": "右手",
        "flag": 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_R_Shoulder": {
        "name": "右肩",
        "parent": "shoulderP_R",
        "tail": "J_Bip_R_UpperArm",
        "display": "右手",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "shoulderC_R": {
        "name": "右肩C",
        "parent": "J_Bip_R_Shoulder",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "J_Bip_R_UpperArm": {
        "name": "右腕",
        "parent": "shoulderC_R",
        "tail": "J_Bip_R_LowerArm",
        "display": "右手",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "arm_twist_R": {
        "name": "右腕捩",
        "parent": "J_Bip_R_UpperArm",
        "tail": None,
        "display": "右手",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800,
    },
    "arm_twist_1R": {
        "name": "右腕捩1",
        "parent": "J_Bip_R_UpperArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "arm_twist_2R": {
        "name": "右腕捩2",
        "parent": "J_Bip_R_UpperArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "arm_twist_3R": {
        "name": "右腕捩3",
        "parent": "J_Bip_R_UpperArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "J_Bip_R_LowerArm": {
        "name": "右ひじ",
        "parent": "arm_twist_R",
        "tail": "J_Bip_R_Hand",
        "display": "右手",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "wrist_twist_R": {
        "name": "右手捩",
        "parent": "J_Bip_R_LowerArm",
        "tail": None,
        "display": "右手",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800,
    },
    "wrist_twist_1R": {
        "name": "右手捩1",
        "parent": "J_Bip_R_LowerArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "wrist_twist_2R": {
        "name": "右手捩2",
        "parent": "J_Bip_R_LowerArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "wrist_twist_3R": {
        "name": "右手捩3",
        "parent": "J_Bip_R_LowerArm",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "J_Bip_R_Hand": {
        "name": "右手首",
        "parent": "wrist_twist_R",
        "tail": None,
        "display": "右手",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Thumb1": {
        "name": "右親指０",
        "parent": "J_Bip_R_Hand",
        "tail": "J_Bip_R_Thumb2",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Thumb2": {
        "name": "右親指１",
        "parent": "J_Bip_R_Thumb1",
        "tail": "J_Bip_R_Thumb3",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Thumb3": {
        "name": "右親指２",
        "parent": "J_Bip_R_Thumb2",
        "tail": "J_Bip_R_Thumb3_end",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Thumb3_end": {
        "name": "右親指先",
        "parent": "J_Bip_R_Thumb3",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "J_Bip_R_Index1": {
        "name": "右人指１",
        "parent": "J_Bip_R_Hand",
        "tail": "J_Bip_R_Index2",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Index2": {
        "name": "右人指２",
        "parent": "J_Bip_R_Index1",
        "tail": "J_Bip_R_Index3",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Index3": {
        "name": "右人指３",
        "parent": "J_Bip_R_Index2",
        "tail": "J_Bip_R_Index3_end",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Index3_end": {
        "name": "右人指先",
        "parent": "J_Bip_R_Index3",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "J_Bip_R_Middle1": {
        "name": "右中指１",
        "parent": "J_Bip_R_Hand",
        "tail": "J_Bip_R_Middle2",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Middle2": {
        "name": "右中指２",
        "parent": "J_Bip_R_Middle1",
        "tail": "J_Bip_R_Middle3",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Middle3": {
        "name": "右中指３",
        "parent": "J_Bip_R_Middle2",
        "tail": "J_Bip_R_Middle3_end",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Middle3_end": {
        "name": "右中指先",
        "parent": "J_Bip_R_Middle3",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "J_Bip_R_Ring1": {
        "name": "右薬指１",
        "parent": "J_Bip_R_Hand",
        "tail": "J_Bip_R_Ring2",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Ring2": {
        "name": "右薬指２",
        "parent": "J_Bip_R_Ring1",
        "tail": "J_Bip_R_Ring3",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Ring3": {
        "name": "右薬指３",
        "parent": "J_Bip_R_Ring2",
        "tail": "J_Bip_R_Ring3_end",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Ring3_end": {
        "name": "右薬指先",
        "parent": "J_Bip_R_Ring3",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "J_Bip_R_Little1": {
        "name": "右小指１",
        "parent": "J_Bip_R_Hand",
        "tail": "J_Bip_R_Little2",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Little2": {
        "name": "右小指２",
        "parent": "J_Bip_R_Little1",
        "tail": "J_Bip_R_Little3",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Little3": {
        "name": "右小指３",
        "parent": "J_Bip_R_Little2",
        "tail": "J_Bip_R_Little3_end",
        "display": "右指",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800,
    },
    "J_Bip_R_Little3_end": {
        "name": "右小指先",
        "parent": "J_Bip_R_Little3",
        "tail": None,
        "display": None,
        "flag": 0x0002,
    },
    "waistCancel_L": {
        "name": "腰キャンセル左",
        "parent": "J_Bip_C_Spine",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "J_Bip_L_UpperLeg": {
        "name": "左足",
        "parent": "waistCancel_L",
        "tail": "J_Bip_L_LowerLeg",
        "display": "左足",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_L_LowerLeg": {
        "name": "左ひざ",
        "parent": "J_Bip_L_UpperLeg",
        "tail": "J_Bip_L_Foot",
        "display": "左足",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_L_Foot": {
        "name": "左足首",
        "parent": "J_Bip_L_LowerLeg",
        "tail": "J_Bip_L_ToeBase",
        "display": "左足",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_L_ToeBase": {
        "name": "左つま先",
        "parent": "J_Bip_L_Foot",
        "tail": None,
        "display": "左足",
        "flag": 0x0002 | 0x0008 | 0x0010,
    },
    "leg_IK_Parent_L": {
        "name": "左足IK親",
        "parent": "Root",
        "tail": "leg_IK_L",
        "display": "左足",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010,
    },
    "leg_IK_L": {
        "name": "左足ＩＫ",
        "parent": "leg_IK_Parent_L",
        "tail": MVector3D(0, 0, 1),
        "display": "左足",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020,
    },
    "toe_IK_L": {
        "name": "左つま先ＩＫ",
        "parent": "leg_IK_L",
        "tail": MVector3D(0, -1, 0),
        "display": "左足",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020,
    },
    "waistCancel_R": {
        "name": "腰キャンセル右",
        "parent": "J_Bip_C_Spine",
        "tail": None,
        "display": None,
        "flag": 0x0002 | 0x0100,
    },
    "J_Bip_R_UpperLeg": {
        "name": "右足",
        "parent": "waistCancel_R",
        "tail": "J_Bip_R_LowerLeg",
        "display": "右足",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_R_LowerLeg": {
        "name": "右ひざ",
        "parent": "J_Bip_R_UpperLeg",
        "tail": "J_Bip_R_Foot",
        "display": "右足",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_R_Foot": {
        "name": "右足首",
        "parent": "J_Bip_R_LowerLeg",
        "tail": "J_Bip_R_ToeBase",
        "display": "右足",
        "flag": 0x0001 | 0x0002 | 0x0008 | 0x0010,
    },
    "J_Bip_R_ToeBase": {
        "name": "右つま先",
        "parent": "J_Bip_R_Foot",
        "tail": None,
        "display": "右足",
        "flag": 0x0002 | 0x0008 | 0x0010,
    },
    "leg_IK_Parent_R": {
        "name": "右足IK親",
        "parent": "Root",
        "tail": "leg_IK_R",
        "display": "右足",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010,
    },
    "leg_IK_R": {
        "name": "右足ＩＫ",
        "parent": "leg_IK_Parent_R",
        "tail": MVector3D(0, 0, 1),
        "display": "右足",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020,
    },
    "toe_IK_R": {
        "name": "右つま先ＩＫ",
        "parent": "leg_IK_R",
        "tail": MVector3D(0, -1, 0),
        "display": "右足",
        "flag": 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020,
    },
    "leg_D_L": {
        "name": "左足D",
        "parent": "waistCancel_L",
        "tail": None,
        "display": "左足",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0100,
    },
    "knee_D_L": {
        "name": "左ひざD",
        "parent": "leg_D_L",
        "tail": None,
        "display": "左足",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0100,
    },
    "ankle_D_L": {
        "name": "左足首D",
        "parent": "knee_D_L",
        "tail": None,
        "display": "左足",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0100,
    },
    "toe_EX_L": {
        "name": "左足先EX",
        "parent": "ankle_D_L",
        "tail": MVector3D(0, 0, -1),
        "display": "左足",
        "flag": 0x0002 | 0x0008 | 0x0010,
    },
    "leg_D_R": {
        "name": "右足D",
        "parent": "waistCancel_R",
        "tail": None,
        "display": "右足",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0100,
    },
    "knee_D_R": {
        "name": "右ひざD",
        "parent": "leg_D_R",
        "tail": None,
        "display": "右足",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0100,
    },
    "ankle_D_R": {
        "name": "右足首D",
        "parent": "knee_D_R",
        "tail": None,
        "display": "右足",
        "flag": 0x0002 | 0x0008 | 0x0010 | 0x0100,
    },
    "toe_EX_R": {
        "name": "右足先EX",
        "parent": "ankle_D_R",
        "tail": MVector3D(0, 0, -1),
        "display": "右足",
        "flag": 0x0002 | 0x0008 | 0x0010,
    },
}

RIGIDBODY_PAIRS = {
    "下半身": {
        "bone": "下半身",
        "english": "J_Bip_C_Spine",
        "group": 0,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "horizonal",
        "range": "all",
        "ratio": MVector3D(0.8, 0.4, 1),
    },
    "上半身": {
        "bone": "上半身",
        "english": "J_Bip_C_Spine2",
        "group": 0,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "horizonal",
        "range": "all",
        "ratio": MVector3D(1, 0.5, 1),
    },
    "上半身2": {
        "bone": "上半身2",
        "english": "J_Bip_C_Chest",
        "group": 0,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "horizonal",
        "range": "all",
        "ratio": MVector3D(0.7, 0.4, 1),
    },
    "上半身3": {
        "bone": "上半身3",
        "english": "J_Bip_C_UpperChest",
        "group": 0,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "horizonal",
        "range": "all",
        "ratio": MVector3D(0.7, 0.4, 1),
    },
    "首": {
        "bone": "首",
        "english": "J_Bip_C_Neck",
        "group": 0,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "vertical",
        "range": "all",
        "ratio": MVector3D(0.6, 0.7, 1),
    },
    "頭": {
        "bone": "頭",
        "english": "J_Bip_C_Head",
        "group": 0,
        "shape": 0,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "lower",
        "ratio": MVector3D(0.6, 1, 1),
    },
    "後頭部": {
        "bone": "頭",
        "english": "J_Bip_C_HeadBack",
        "group": 0,
        "shape": 0,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "upper",
        "ratio": MVector3D(1, 1, 1),
    },
    "左胸": {
        "bone": "左胸",
        "english": "J_Sec_L_Bust1",
        "group": 0,
        "shape": 0,
        "no_collision_group": [0, 1, 2],
        "direction": "vertical",
        "range": "all",
        "ratio": MVector3D(1.6, 1, 1),
    },
    "右胸": {
        "bone": "右胸",
        "english": "J_Sec_R_Bust1",
        "group": 0,
        "shape": 0,
        "no_collision_group": [0, 1, 2],
        "direction": "vertical",
        "range": "all",
        "ratio": MVector3D(1.6, 1, 1),
    },
    "左肩": {
        "bone": "左肩",
        "english": "J_Bip_L_Shoulder",
        "group": 2,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "vertical",
        "range": "all",
        "ratio": MVector3D(0.5, 0.9, 1),
    },
    "左腕": {
        "bone": "左腕",
        "english": "J_Bip_L_UpperArm",
        "group": 2,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "vertical",
        "range": "all",
    },
    "左ひじ": {
        "bone": "左ひじ",
        "english": "J_Bip_L_LowerArm",
        "group": 2,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "vertical",
        "range": "all",
    },
    "左手首": {
        "bone": "左手首",
        "english": "J_Bip_L_Hand",
        "group": 2,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "vertical",
        "range": "all",
        "ratio": MVector3D(0.8, 0.6, 1),
    },
    "右肩": {
        "bone": "右肩",
        "english": "J_Bip_R_Shoulder",
        "group": 2,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "vertical",
        "range": "all",
        "ratio": MVector3D(0.5, 0.9, 1),
    },
    "右腕": {
        "bone": "右腕",
        "english": "J_Bip_R_UpperArm",
        "group": 2,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "vertical",
        "range": "all",
    },
    "右ひじ": {
        "bone": "右ひじ",
        "english": "J_Bip_R_LowerArm",
        "group": 2,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "vertical",
        "range": "all",
    },
    "右手首": {
        "bone": "右手首",
        "english": "J_Bip_R_Hand",
        "group": 2,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "vertical",
        "range": "all",
        "ratio": MVector3D(0.8, 0.6, 1),
    },
    "左尻": {
        "bone": "左足",
        "english": "J_Bip_L_Hip",
        "group": 1,
        "shape": 0,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "all",
    },
    "左太もも": {
        "bone": "左足",
        "range_bone": "下半身",
        "english": "J_Bip_L_UpperLegUpper",
        "group": 1,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "lower",
        "ratio": MVector3D(0.9, 1, 1),
    },
    "左足": {
        "bone": "左足",
        "english": "J_Bip_L_UpperLegLower",
        "group": 1,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "upper",
    },
    "左ひざ": {
        "bone": "左ひざ",
        "english": "J_Bip_L_LowerLegUpper",
        "group": 1,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "lower",
    },
    "左すね": {
        "bone": "左ひざ",
        "english": "J_Bip_L_LowerLegLower",
        "group": 1,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "upper",
    },
    "左足首": {
        "bone": "左足首",
        "english": "J_Bip_L_Foot",
        "group": 1,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "all",
        "ratio": MVector3D(1, 0.7, 1),
    },
    "右尻": {
        "bone": "右足",
        "range_bone": "下半身",
        "english": "J_Bip_R_Hip",
        "group": 1,
        "shape": 0,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "all",
    },
    "右太もも": {
        "bone": "右足",
        "english": "J_Bip_R_UpperLegUpper",
        "group": 1,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "lower",
        "ratio": MVector3D(0.9, 1, 1),
    },
    "右足": {
        "bone": "右足",
        "english": "J_Bip_R_UpperLegLower",
        "group": 1,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "upper",
    },
    "右ひざ": {
        "bone": "右ひざ",
        "english": "J_Bip_R_LowerLegUpper",
        "group": 1,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "lower",
    },
    "右すね": {
        "bone": "右ひざ",
        "english": "J_Bip_R_LowerLegLower",
        "group": 1,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "upper",
    },
    "右足首": {
        "bone": "右足首",
        "english": "J_Bip_R_Foot",
        "group": 1,
        "shape": 2,
        "no_collision_group": [0, 1, 2],
        "direction": "reverse",
        "range": "all",
        "ratio": MVector3D(1, 0.7, 1),
    },
}

MORPH_SYSTEM = 0
MORPH_EYEBROW = 1
MORPH_EYE = 2
MORPH_LIP = 3
MORPH_OTHER = 4

DEFINED_MORPH_NAMES = [
    "Neutral",
    "A",
    "I",
    "U",
    "E",
    "O",
    "Blink",
    "Blink_L",
    "Blink_R",
    "Angry",
    "Fun",
    "Joy",
    "Sorrow",
    "Surprised",
]

MORPH_PAIRS = {
    "Fcl_BRW_Fun_R": {"name": "にこり右", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Fun"},
    "Fcl_BRW_Fun_L": {"name": "にこり左", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Fun"},
    "Fcl_BRW_Fun": {"name": "にこり", "panel": MORPH_EYEBROW},
    "Fcl_BRW_Joy_R": {"name": "にこり2右", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Joy"},
    "Fcl_BRW_Joy_L": {"name": "にこり2左", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Joy"},
    "Fcl_BRW_Joy": {"name": "にこり2", "panel": MORPH_EYEBROW},
    "Fcl_BRW_Sorrow_R": {"name": "困る右", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Sorrow"},
    "Fcl_BRW_Sorrow_L": {"name": "困る左", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Sorrow"},
    "Fcl_BRW_Sorrow": {"name": "困る", "panel": MORPH_EYEBROW},
    "Fcl_BRW_Angry_R": {"name": "怒り右", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Angry"},
    "Fcl_BRW_Angry_L": {"name": "怒り左", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Angry"},
    "Fcl_BRW_Angry": {"name": "怒り", "panel": MORPH_EYEBROW},
    "Fcl_BRW_Surprised_R": {"name": "驚き右", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Surprised"},
    "Fcl_BRW_Surprised_L": {"name": "驚き左", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Surprised"},
    "Fcl_BRW_Surprised": {"name": "驚き", "panel": MORPH_EYEBROW},
    "brow_Below_R": {"name": "下右", "panel": MORPH_EYEBROW, "creates": ["FaceBrow"]},
    "brow_Below_L": {"name": "下左", "panel": MORPH_EYEBROW, "creates": ["FaceBrow"]},
    "brow_Below": {"name": "下", "panel": MORPH_EYEBROW, "binds": ["brow_Below_R", "brow_Below_L"]},
    "brow_Abobe_R": {"name": "上右", "panel": MORPH_EYEBROW, "creates": ["FaceBrow"]},
    "brow_Abobe_L": {"name": "上左", "panel": MORPH_EYEBROW, "creates": ["FaceBrow"]},
    "brow_Abobe": {"name": "上", "panel": MORPH_EYEBROW, "binds": ["brow_Abobe_R", "brow_Abobe_L"]},
    "brow_Left_R": {"name": "右眉左", "panel": MORPH_EYEBROW, "creates": ["FaceBrow"]},
    "brow_Left_L": {"name": "左眉左", "panel": MORPH_EYEBROW, "creates": ["FaceBrow"]},
    "brow_Left": {"name": "眉左", "panel": MORPH_EYEBROW, "binds": ["brow_Left_R", "brow_Left_L"]},
    "brow_Right_R": {"name": "右眉右", "panel": MORPH_EYEBROW, "creates": ["FaceBrow"]},
    "brow_Right_L": {"name": "左眉右", "panel": MORPH_EYEBROW, "creates": ["FaceBrow"]},
    "brow_Right": {"name": "眉右", "panel": MORPH_EYEBROW, "binds": ["brow_Right_R", "brow_Right_L"]},
    "brow_Front_R": {"name": "右眉手前", "panel": MORPH_EYEBROW, "creates": ["FaceBrow"]},
    "brow_Front_L": {"name": "左眉手前", "panel": MORPH_EYEBROW, "creates": ["FaceBrow"]},
    "brow_Front": {"name": "眉手前", "panel": MORPH_EYEBROW, "binds": ["brow_Front_R", "brow_Front_L"]},
    "brow_Serious_R": {
        "name": "真面目右",
        "panel": MORPH_EYEBROW,
        "binds": ["Fcl_BRW_Angry_R", "brow_Below_R"],
        "ratios": [0.25, 0.7],
    },
    "brow_Serious_L": {
        "name": "真面目左",
        "panel": MORPH_EYEBROW,
        "binds": ["Fcl_BRW_Angry_L", "brow_Below_L"],
        "ratios": [0.25, 0.7],
    },
    "brow_Serious": {
        "name": "真面目",
        "panel": MORPH_EYEBROW,
        "binds": ["Fcl_BRW_Angry_R", "brow_Below_R", "Fcl_BRW_Angry_L", "brow_Below_L"],
        "ratios": [0.25, 0.7, 0.25, 0.7],
    },
    "brow_Frown_R": {
        "name": "ひそめ右",
        "panel": MORPH_EYEBROW,
        "binds": ["Fcl_BRW_Angry_R", "Fcl_BRW_Sorrow_R", "brow_Right_R"],
        "ratios": [0.5, 0.5, 0.3],
    },
    "brow_Frown_L": {
        "name": "ひそめ左",
        "panel": MORPH_EYEBROW,
        "binds": ["Fcl_BRW_Angry_L", "Fcl_BRW_Sorrow_L", "brow_Left_L"],
        "ratios": [0.5, 0.5, 0.3],
    },
    "brow_Frown": {
        "name": "ひそめ",
        "panel": MORPH_EYEBROW,
        "binds": [
            "Fcl_BRW_Angry_R",
            "Fcl_BRW_Sorrow_R",
            "brow_Right_R",
            "Fcl_BRW_Angry_L",
            "Fcl_BRW_Sorrow_L",
            "brow_Left_L",
        ],
        "ratios": [0.5, 0.5, 0.3, 0.5, 0.5, 0.3],
    },
    "browInnerUp_R": {"name": "ひそめる2右", "panel": MORPH_EYEBROW, "split": "browInnerUp"},
    "browInnerUp_L": {"name": "ひそめる2左", "panel": MORPH_EYEBROW, "split": "browInnerUp"},
    "browInnerUp": {"name": "ひそめる2", "panel": MORPH_EYEBROW},
    "browDownRight": {"name": "真面目2右", "panel": MORPH_EYEBROW},
    "browDownLeft": {"name": "真面目2左", "panel": MORPH_EYEBROW},
    "browDown": {"name": "真面目2", "panel": MORPH_EYEBROW, "binds": ["browDownRight", "browDownLeft"]},
    "browOuterUpRight": {"name": "はんっ右", "panel": MORPH_EYEBROW},
    "browOuterUpLeft": {"name": "はんっ左", "panel": MORPH_EYEBROW},
    "browOuter": {"name": "はんっ", "panel": MORPH_EYEBROW, "binds": ["browOuterUpRight", "browOuterUpLeft"]},
    "Fcl_EYE_Surprised_R": {"name": "びっくり右", "panel": MORPH_EYE, "split": "Fcl_EYE_Surprised"},
    "Fcl_EYE_Surprised_L": {"name": "びっくり左", "panel": MORPH_EYE, "split": "Fcl_EYE_Surprised"},
    "Fcl_EYE_Surprised": {"name": "びっくり", "panel": MORPH_EYE},
    "eye_Small_R": {"name": "瞳小右", "panel": MORPH_EYE, "creates": ["EyeIris", "EyeHighlight"]},
    "eye_Small_L": {"name": "瞳小左", "panel": MORPH_EYE, "creates": ["EyeIris", "EyeHighlight"]},
    "eye_Small": {"name": "瞳小", "panel": MORPH_EYE, "binds": ["eye_Small_R", "eye_Small_L"]},
    "eye_Big_R": {"name": "瞳大右", "panel": MORPH_EYE, "creates": ["EyeIris", "EyeHighlight"]},
    "eye_Big_L": {"name": "瞳大左", "panel": MORPH_EYE, "creates": ["EyeIris", "EyeHighlight"]},
    "eye_Big": {"name": "瞳大", "panel": MORPH_EYE, "binds": ["eye_Big_R", "eye_Big_L"]},
    "Fcl_EYE_Close_R": {"name": "ｳｨﾝｸ２右", "panel": MORPH_EYE},
    "Fcl_EYE_Close_R_Bone": {
        "name": "ｳｨﾝｸ２右ボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "右目光",
        ],
        "move_ratios": [
            MVector3D(0, 0, -0.015),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(-12, 0, 0),
        ],
    },
    "Fcl_EYE_Close_R_Group": {
        "name": "ｳｨﾝｸ２右連動",
        "panel": MORPH_EYE,
        "binds": [
            "brow_Below_R",
            "Fcl_EYE_Close_R",
            "eye_Small_R",
            "Fcl_EYE_Close_R_Bone",
            "brow_Front_R",
            "Fcl_BRW_Sorrow_R",
        ],
        "ratios": [0.2, 1, 0.3, 1, 0.1, 0.2],
    },
    "Fcl_EYE_Close_L": {"name": "ウィンク２", "panel": MORPH_EYE},
    "Fcl_EYE_Close_L_Bone": {
        "name": "ウィンク２ボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "左目光",
        ],
        "move_ratios": [
            MVector3D(0, 0, -0.015),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(-12, 0, 0),
        ],
    },
    "Fcl_EYE_Close_L_Group": {
        "name": "ウィンク２連動",
        "panel": MORPH_EYE,
        "binds": [
            "brow_Below_L",
            "Fcl_EYE_Close_L",
            "eye_Small_L",
            "Fcl_EYE_Close_L_Bone",
            "brow_Front_L",
            "Fcl_BRW_Sorrow_L",
        ],
        "ratios": [0.2, 1, 0.3, 1, 0.1, 0.2],
    },
    "Fcl_EYE_Close": {"name": "まばたき", "panel": MORPH_EYE},
    "Fcl_EYE_Close_Group": {
        "name": "まばたき連動",
        "panel": MORPH_EYE,
        "binds": [
            "brow_Below_R",
            "Fcl_EYE_Close_R",
            "eye_Small_R",
            "Fcl_EYE_Close_R_Bone",
            "brow_Front_R",
            "Fcl_BRW_Sorrow_R",
            "brow_Below_L",
            "Fcl_EYE_Close_L",
            "eye_Small_L",
            "Fcl_EYE_Close_L_Bone",
            "brow_Front_L",
            "Fcl_BRW_Sorrow_L",
        ],
        "ratios": [0.2, 1, 0.3, 1, 0.1, 0.2, 0.2, 1, 0.3, 1, 0.1, 0.2],
    },
    "Fcl_EYE_Joy_R": {"name": "ウィンク右", "panel": MORPH_EYE},
    "Fcl_EYE_Joy_R_Bone": {
        "name": "ウィンク右ボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "右目光",
        ],
        "move_ratios": [
            MVector3D(0, 0, 0.025),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(8, 0, 0),
        ],
    },
    "Fcl_EYE_Joy_R_Group": {
        "name": "ウィンク右連動",
        "panel": MORPH_EYE,
        "binds": [
            "brow_Below_R",
            "Fcl_EYE_Joy_R",
            "eye_Small_R",
            "Fcl_EYE_Joy_R_Bone",
            "brow_Front_R",
            "Fcl_BRW_Fun_R",
        ],
        "ratios": [0.5, 1, 0.3, 1, 0.1, 0.5],
    },
    "Fcl_EYE_Joy_L": {"name": "ウィンク", "panel": MORPH_EYE},
    "Fcl_EYE_Joy_L_Bone": {
        "name": "ウィンクボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "左目光",
        ],
        "move_ratios": [
            MVector3D(0, 0, 0.025),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(8, 0, 0),
        ],
    },
    "Fcl_EYE_Joy_L_Group": {
        "name": "ウィンク連動",
        "panel": MORPH_EYE,
        "binds": [
            "brow_Below_L",
            "Fcl_EYE_Joy_L",
            "eye_Small_L",
            "Fcl_EYE_Joy_L_Bone",
            "brow_Front_L",
            "Fcl_BRW_Fun_L",
        ],
        "ratios": [0.5, 1, 0.3, 1, 0.1, 0.5],
    },
    "Fcl_EYE_Joy": {"name": "笑い", "panel": MORPH_EYE},
    "Fcl_EYE_Joy_Group": {
        "name": "笑い連動",
        "panel": MORPH_EYE,
        "binds": [
            "brow_Below_R",
            "Fcl_EYE_Joy_R",
            "eye_Small_R",
            "Fcl_EYE_Joy_R_Bone",
            "brow_Front_R",
            "Fcl_BRW_Fun_R",
            "brow_Below_L",
            "Fcl_EYE_Joy_L",
            "eye_Small_L",
            "Fcl_EYE_Joy_L_Bone",
            "brow_Front_L",
            "Fcl_BRW_Fun_L",
        ],
        "ratios": [0.5, 1, 0.3, 1, 0.1, 0.5, 0.5, 1, 0.3, 1, 0.1, 0.5],
    },
    "Fcl_EYE_Fun_R": {"name": "目を細める右", "panel": MORPH_EYE, "split": "Fcl_EYE_Fun"},
    "Fcl_EYE_Fun_L": {"name": "目を細める左", "panel": MORPH_EYE, "split": "Fcl_EYE_Fun"},
    "Fcl_EYE_Fun": {"name": "目を細める", "panel": MORPH_EYE},
    "raiseEyelid_R": {"name": "下瞼上げ右", "panel": MORPH_EYE, "split": "Fcl_EYE_Fun_R"},
    "raiseEyelid_L": {"name": "下瞼上げ左", "panel": MORPH_EYE, "split": "Fcl_EYE_Fun_L"},
    "raiseEyelid": {"name": "下瞼上げ", "panel": MORPH_EYE, "binds": ["raiseEyelid_R", "raiseEyelid_L"]},
    "eyeSquintRight": {"name": "にんまり右", "panel": MORPH_EYE},
    "eyeSquintLeft": {"name": "にんまり左", "panel": MORPH_EYE},
    "eyeSquint": {"name": "にんまり", "panel": MORPH_EYE, "binds": ["eyeSquintRight", "eyeSquintLeft"]},
    "Fcl_EYE_Angry_R": {"name": "ｷﾘｯ右", "panel": MORPH_EYE, "split": "Fcl_EYE_Angry"},
    "Fcl_EYE_Angry_L": {"name": "ｷﾘｯ左", "panel": MORPH_EYE, "split": "Fcl_EYE_Angry"},
    "Fcl_EYE_Angry": {"name": "ｷﾘｯ", "panel": MORPH_EYE},
    "noseSneerRight": {"name": "ｷﾘｯ2右", "panel": MORPH_EYE},
    "noseSneerLeft": {"name": "ｷﾘｯ2左", "panel": MORPH_EYE},
    "noseSneer": {"name": "ｷﾘｯ2", "panel": MORPH_EYE, "binds": ["noseSneerRight", "noseSneerLeft"]},
    "Fcl_EYE_Sorrow_R": {"name": "じと目右", "panel": MORPH_EYE, "split": "Fcl_EYE_Sorrow"},
    "Fcl_EYE_Sorrow_L": {"name": "じと目左", "panel": MORPH_EYE, "split": "Fcl_EYE_Sorrow"},
    "Fcl_EYE_Sorrow": {"name": "じと目", "panel": MORPH_EYE},
    "Fcl_EYE_Spread_R": {"name": "上瞼↑右", "panel": MORPH_EYE, "split": "Fcl_EYE_Spread"},
    "Fcl_EYE_Spread_L": {"name": "上瞼↑左", "panel": MORPH_EYE, "split": "Fcl_EYE_Spread"},
    "Fcl_EYE_Spread": {"name": "上瞼↑", "panel": MORPH_EYE},
    "eye_Nanu_R": {
        "name": "なぬ！右",
        "panel": MORPH_EYE,
        "binds": ["Fcl_EYE_Surprised_R", "Fcl_EYE_Angry_R"],
        "ratios": [1, 1],
    },
    "eye_Nanu_L": {
        "name": "なぬ！左",
        "panel": MORPH_EYE,
        "binds": ["Fcl_EYE_Surprised_L", "Fcl_EYE_Angry_L"],
        "ratios": [1, 1],
    },
    "eye_Nanu": {
        "name": "なぬ！",
        "panel": MORPH_EYE,
        "binds": ["Fcl_EYE_Surprised_R", "Fcl_EYE_Angry_R", "Fcl_EYE_Surprised_L", "Fcl_EYE_Angry_L"],
        "ratios": [1, 1, 1, 1],
    },
    "eye_Hide_Vertex": {
        "name": "目隠し頂点",
        "panel": MORPH_SYSTEM,
        "creates": ["EyeWhite"],
        "hides": ["Eyeline", "Eyelash"],
    },
    "eye_Hau_Material": {
        "name": "はぅ材質",
        "panel": MORPH_SYSTEM,
        "material": "eye_hau",
        "hides": ["EyeWhite", "Eyeline", "Eyelash"],
    },
    "eye_Hau": {
        "name": "はぅ",
        "panel": MORPH_EYE,
        "binds": ["eye_Hau_Material", "eye_Hide_Vertex"],
    },
    "eye_Hachume_Material": {
        "name": "はちゅ目材質",
        "panel": MORPH_SYSTEM,
        "material": "eye_hachume",
        "hides": ["EyeWhite", "Eyeline", "Eyelash"],
    },
    "eye_Hachume": {
        "name": "はちゅ目",
        "panel": MORPH_EYE,
        "binds": ["eye_Hachume_Material", "eye_Hide_Vertex"],
    },
    "eye_Nagomi_Material": {
        "name": "なごみ材質",
        "panel": MORPH_SYSTEM,
        "material": "eye_nagomi",
        "hides": ["EyeWhite", "Eyeline", "Eyelash"],
    },
    "eye_Nagomi": {
        "name": "なごみ",
        "panel": MORPH_EYE,
        "binds": ["eye_Nagomi_Material", "eye_Hide_Vertex"],
    },
    "eye_Star_Material": {"name": "星目材質", "panel": MORPH_SYSTEM, "material": "eye_star"},
    "eye_Heart_Material": {"name": "はぁと材質", "panel": MORPH_SYSTEM, "material": "eye_heart"},
    "eye_Star": {"name": "星目", "panel": MORPH_EYE, "binds": ["Fcl_EYE_Highlight_Hide", "eye_Star_Material"]},
    "eye_Heart": {"name": "はぁと", "panel": MORPH_EYE, "binds": ["Fcl_EYE_Highlight_Hide", "eye_Heart_Material"]},
    "Fcl_EYE_Natural": {"name": "ナチュラル", "panel": MORPH_EYE},
    "eyeWideRight": {"name": "びっくり2右", "panel": MORPH_EYE},
    "eyeWideLeft": {"name": "びっくり2左", "panel": MORPH_EYE},
    "eyeWide": {"name": "びっくり2", "panel": MORPH_EYE, "binds": ["eyeSquintRight", "eyeSquintLeft"]},
    "eyeLookUpRight": {"name": "目上右", "panel": MORPH_EYE},
    "eyeLookUpLeft": {"name": "目上左", "panel": MORPH_EYE},
    "eyeLookUp": {"name": "目上", "panel": MORPH_EYE, "binds": ["eyeLookUpRight", "eyeLookUpLeft"]},
    "eyeLookDownRight": {"name": "目下右", "panel": MORPH_EYE},
    "eyeLookDownLeft": {"name": "目下左", "panel": MORPH_EYE},
    "eyeLookDown": {"name": "目下", "panel": MORPH_EYE, "binds": ["eyeLookDownRight", "eyeLookDownLeft"]},
    "eyeLookInRight": {"name": "目頭広右", "panel": MORPH_EYE},
    "eyeLookInLeft": {"name": "目頭広左", "panel": MORPH_EYE},
    "eyeLookIn": {"name": "目頭広", "panel": MORPH_EYE, "binds": ["eyeLookInRight", "eyeLookInLeft"]},
    "eyeLookOutLeft": {"name": "目尻広右", "panel": MORPH_EYE},
    "eyeLookOutRight": {"name": "目尻広左", "panel": MORPH_EYE},
    "eyeLookOut": {"name": "目尻広", "panel": MORPH_EYE, "binds": ["eyeLookOutRight", "eyeLookOutLeft"]},
    # "eyeBlinkLeft": {"name": "", "panel": MORPH_EYE},
    # "eyeBlinkRight": {"name": "", "panel": MORPH_EYE},
    "_eyeIrisMoveBack_R": {"name": "瞳小2右", "panel": MORPH_EYE},
    "_eyeIrisMoveBack_L": {"name": "瞳小2左", "panel": MORPH_EYE},
    "_eyeIrisMoveBack": {"name": "瞳小2", "panel": MORPH_EYE, "binds": ["_eyeIrisMoveBack_R", "_eyeIrisMoveBack_L"]},
    "_eyeSquint+LowerUp_R": {"name": "下瞼上げ2右", "panel": MORPH_EYE},
    "_eyeSquint+LowerUp_L": {"name": "下瞼上げ2左", "panel": MORPH_EYE},
    "_eyeSquint+LowerUp": {
        "name": "下瞼上げ2",
        "panel": MORPH_EYE,
        "binds": ["_eyeSquint+LowerUp_R", "_eyeSquint+LowerUp_L"],
    },
    "Fcl_EYE_Iris_Hide": {"name": "白目", "panel": MORPH_EYE},
    "Fcl_EYE_Iris_Hide_R": {"name": "白目右", "panel": MORPH_EYE, "split": "Fcl_EYE_Iris_Hide"},
    "Fcl_EYE_Iris_Hide_L": {"name": "白目左", "panel": MORPH_EYE, "split": "Fcl_EYE_Iris_Hide"},
    "Fcl_EYE_Highlight_Hide": {"name": "ハイライトなし", "panel": MORPH_EYE},
    "Fcl_EYE_Highlight_Hide_R": {"name": "ハイライトなし右", "panel": MORPH_EYE, "split": "Fcl_EYE_Highlight_Hide"},
    "Fcl_EYE_Highlight_Hide_L": {"name": "ハイライトなし左", "panel": MORPH_EYE, "split": "Fcl_EYE_Highlight_Hide"},
    "Fcl_MTH_A": {"name": "あ頂点", "panel": MORPH_SYSTEM},
    "Fcl_MTH_A_Bone": {
        "name": "あボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "舌1",
            "舌2",
            "舌3",
        ],
        "move_ratios": [
            MVector3D(),
            MVector3D(),
            MVector3D(),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(-16, 0, 0),
            MQuaternion.fromEulerAngles(-16, 0, 0),
            MQuaternion.fromEulerAngles(-10, 0, 0),
        ],
    },
    "Fcl_MTH_A_Group": {"name": "あ", "panel": MORPH_LIP, "binds": ["Fcl_MTH_A", "Fcl_MTH_A_Bone"]},
    "Fcl_MTH_I": {"name": "い頂点", "panel": MORPH_SYSTEM},
    "Fcl_MTH_I_Bone": {
        "name": "いボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "舌1",
            "舌2",
            "舌3",
        ],
        "move_ratios": [
            MVector3D(),
            MVector3D(),
            MVector3D(),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(-6, 0, 0),
            MQuaternion.fromEulerAngles(-6, 0, 0),
            MQuaternion.fromEulerAngles(-3, 0, 0),
        ],
    },
    "Fcl_MTH_I_Group": {"name": "い", "panel": MORPH_LIP, "binds": ["Fcl_MTH_I", "Fcl_MTH_I_Bone"]},
    "Fcl_MTH_U": {"name": "う頂点", "panel": MORPH_SYSTEM},
    "Fcl_MTH_U_Bone": {
        "name": "うボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "舌1",
            "舌2",
            "舌3",
        ],
        "move_ratios": [
            MVector3D(),
            MVector3D(),
            MVector3D(),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(-16, 0, 0),
            MQuaternion.fromEulerAngles(-16, 0, 0),
            MQuaternion.fromEulerAngles(-10, 0, 0),
        ],
    },
    "Fcl_MTH_U_Group": {"name": "う", "panel": MORPH_LIP, "binds": ["Fcl_MTH_U", "Fcl_MTH_U_Bone"]},
    "Fcl_MTH_E": {"name": "え頂点", "panel": MORPH_SYSTEM},
    "Fcl_MTH_E_Bone": {
        "name": "えボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "舌1",
            "舌2",
            "舌3",
        ],
        "move_ratios": [
            MVector3D(),
            MVector3D(),
            MVector3D(),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(-6, 0, 0),
            MQuaternion.fromEulerAngles(-6, 0, 0),
            MQuaternion.fromEulerAngles(-3, 0, 0),
        ],
    },
    "Fcl_MTH_E_Group": {"name": "え", "panel": MORPH_LIP, "binds": ["Fcl_MTH_E", "Fcl_MTH_E_Bone"]},
    "Fcl_MTH_O": {"name": "お頂点", "panel": MORPH_SYSTEM},
    "Fcl_MTH_O_Bone": {
        "name": "おボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "舌1",
            "舌2",
            "舌3",
        ],
        "move_ratios": [
            MVector3D(),
            MVector3D(),
            MVector3D(),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(-20, 0, 0),
            MQuaternion.fromEulerAngles(-18, 0, 0),
            MQuaternion.fromEulerAngles(-12, 0, 0),
        ],
    },
    "Fcl_MTH_O_Group": {"name": "お", "panel": MORPH_LIP, "binds": ["Fcl_MTH_O", "Fcl_MTH_O_Bone"]},
    "Fcl_MTH_Neutral": {"name": "ん", "panel": MORPH_LIP},
    "Fcl_MTH_Close": {"name": "一文字", "panel": MORPH_LIP},
    "Fcl_MTH_Up": {"name": "口上", "panel": MORPH_LIP},
    "Fcl_MTH_Down": {"name": "口下", "panel": MORPH_LIP},
    "Fcl_MTH_Angry_R": {"name": "Λ右", "panel": MORPH_LIP, "split": "Fcl_MTH_Angry"},
    "Fcl_MTH_Angry_L": {"name": "Λ左", "panel": MORPH_LIP, "split": "Fcl_MTH_Angry"},
    "Fcl_MTH_Angry": {"name": "Λ", "panel": MORPH_LIP},
    "Fcl_MTH_Sage_R": {
        "name": "口角下げ右",
        "panel": MORPH_LIP,
        "binds": ["Fcl_MTH_Angry_R", "Fcl_MTH_Large"],
        "ratios": [1, 0.5],
    },
    "Fcl_MTH_Sage_L": {
        "name": "口角下げ左",
        "panel": MORPH_LIP,
        "binds": ["Fcl_MTH_Angry_L", "Fcl_MTH_Large"],
        "ratios": [1, 0.5],
    },
    "Fcl_MTH_Sage": {
        "name": "口角下げ",
        "panel": MORPH_LIP,
        "binds": ["Fcl_MTH_Angry", "Fcl_MTH_Large"],
        "ratios": [1, 0.5],
    },
    "Fcl_MTH_Small": {"name": "うー", "panel": MORPH_LIP},
    "Fcl_MTH_Large": {"name": "口横広げ", "panel": MORPH_LIP},
    "Fcl_MTH_Fun_R": {"name": "にっこり右", "panel": MORPH_LIP, "split": "Fcl_MTH_Fun"},
    "Fcl_MTH_Fun_L": {"name": "にっこり左", "panel": MORPH_LIP, "split": "Fcl_MTH_Fun"},
    "Fcl_MTH_Fun": {"name": "にっこり", "panel": MORPH_LIP},
    "Fcl_MTH_Niko_R": {
        "name": "にこ右",
        "panel": MORPH_LIP,
        "binds": ["Fcl_MTH_Fun_R", "Fcl_MTH_Large"],
        "ratios": [1, -0.3],
    },
    "Fcl_MTH_Niko_L": {
        "name": "にこ左",
        "panel": MORPH_LIP,
        "binds": ["Fcl_MTH_Fun_L", "Fcl_MTH_Large"],
        "ratios": [1, -0.3],
    },
    "Fcl_MTH_Niko": {
        "name": "にこ",
        "panel": MORPH_LIP,
        "binds": ["Fcl_MTH_Fun_R", "Fcl_MTH_Fun_L", "Fcl_MTH_Large"],
        "ratios": [0.5, 0.5, -0.3],
    },
    "Fcl_MTH_Joy": {"name": "ワ頂点", "panel": MORPH_SYSTEM},
    "Fcl_MTH_Joy_Bone": {
        "name": "ワボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "舌1",
            "舌2",
            "舌3",
            "舌4",
        ],
        "move_ratios": [
            MVector3D(),
            MVector3D(),
            MVector3D(),
            MVector3D(),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(-24, 0, 0),
            MQuaternion.fromEulerAngles(-24, 0, 0),
            MQuaternion.fromEulerAngles(16, 0, 0),
            MQuaternion.fromEulerAngles(28, 0, 0),
        ],
    },
    "Fcl_MTH_Joy_Group": {"name": "ワ", "panel": MORPH_LIP, "binds": ["Fcl_MTH_Joy", "Fcl_MTH_Joy_Bone"]},
    "Fcl_MTH_Sorrow": {"name": "▲頂点", "panel": MORPH_SYSTEM},
    "Fcl_MTH_Sorrow_Bone": {
        "name": "▲ボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "舌1",
            "舌2",
            "舌3",
        ],
        "move_ratios": [
            MVector3D(),
            MVector3D(),
            MVector3D(),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(-6, 0, 0),
            MQuaternion.fromEulerAngles(-6, 0, 0),
            MQuaternion.fromEulerAngles(-3, 0, 0),
        ],
    },
    "Fcl_MTH_Sorrow_Group": {"name": "▲", "panel": MORPH_LIP, "binds": ["Fcl_MTH_Sorrow", "Fcl_MTH_Sorrow_Bone"]},
    "Fcl_MTH_Surprised": {"name": "わー頂点", "panel": MORPH_SYSTEM},
    "Fcl_MTH_Surprised_Bone": {
        "name": "わーボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "舌1",
            "舌2",
            "舌3",
            "舌4",
        ],
        "move_ratios": [
            MVector3D(),
            MVector3D(),
            MVector3D(),
            MVector3D(),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(-24, 0, 0),
            MQuaternion.fromEulerAngles(-24, 0, 0),
            MQuaternion.fromEulerAngles(16, 0, 0),
            MQuaternion.fromEulerAngles(28, 0, 0),
        ],
    },
    "Fcl_MTH_Surprised_Group": {
        "name": "わー",
        "panel": MORPH_LIP,
        "binds": ["Fcl_MTH_Surprised", "Fcl_MTH_Surprised_Bone"],
    },
    "Fcl_MTH_tongueOut": {
        "name": "べーボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "舌1",
            "舌2",
            "舌3",
        ],
        "move_ratios": [
            MVector3D(),
            MVector3D(0, 0, -0.24),
            MVector3D(),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(-9, 0, 0),
            MQuaternion.fromEulerAngles(-13.2, 0, 0),
            MQuaternion.fromEulerAngles(-23.2, 0, 0),
        ],
    },
    "Fcl_MTH_tongueOut_Group": {
        "name": "べー",
        "panel": MORPH_LIP,
        "binds": ["Fcl_MTH_A", "Fcl_MTH_I", "Fcl_MTH_tongueOut"],
        "ratios": [0.12, 0.56, 1],
    },
    "Fcl_MTH_tongueUp": {
        "name": "ぺろりボーン",
        "panel": MORPH_SYSTEM,
        "bone": [
            "舌1",
            "舌2",
            "舌3",
            "舌4",
        ],
        "move_ratios": [
            MVector3D(),
            MVector3D(0, -0.03, -0.18),
            MVector3D(),
            MVector3D(),
        ],
        "rotate_ratios": [
            MQuaternion.fromEulerAngles(0, -5, 0),
            MQuaternion.fromEulerAngles(33, -16, -4),
            MQuaternion.fromEulerAngles(15, 3.6, -1),
            MQuaternion.fromEulerAngles(20, 0, 0),
        ],
    },
    "Fcl_MTH_tongueUp_Group": {
        "name": "ぺろり",
        "panel": MORPH_LIP,
        "binds": ["Fcl_MTH_A", "Fcl_MTH_Fun", "Fcl_MTH_tongueUp"],
        "ratios": [0.12, 0.54, 1],
    },
    "jawOpen": {"name": "あああ", "panel": MORPH_LIP},
    "jawForward": {"name": "顎前", "panel": MORPH_LIP},
    "jawLeft": {"name": "顎左", "panel": MORPH_LIP},
    "jawRight": {"name": "顎右", "panel": MORPH_LIP},
    "mouthFunnel": {"name": "んむー", "panel": MORPH_LIP},
    "mouthPucker": {"name": "うー", "panel": MORPH_LIP},
    "mouthLeft": {"name": "口左", "panel": MORPH_LIP},
    "mouthRight": {"name": "口右", "panel": MORPH_LIP},
    "mouthRollUpper": {"name": "上唇んむー", "panel": MORPH_LIP},
    "mouthRollLower": {"name": "下唇んむー", "panel": MORPH_LIP},
    "mouthRoll": {"name": "んむー", "panel": MORPH_LIP, "binds": ["mouthRollUpper", "mouthRollLower"]},
    "mouthShrugUpper": {"name": "上唇むむ", "panel": MORPH_LIP},
    "mouthShrugLower": {"name": "下唇むむ", "panel": MORPH_LIP},
    "mouthShrug": {"name": "むむ", "panel": MORPH_LIP, "binds": ["mouthShrugUpper", "mouthShrugLower"]},
    # "mouthClose": {"name": "", "panel": MORPH_LIP},
    "mouthDimpleRight": {"name": "口幅広右", "panel": MORPH_LIP},
    "mouthDimpleLeft": {"name": "口幅広左", "panel": MORPH_LIP},
    "mouthDimple": {"name": "口幅広", "panel": MORPH_LIP, "binds": ["mouthDimpleRight", "mouthDimpleLeft"]},
    "mouthPressRight": {"name": "薄笑い右", "panel": MORPH_LIP},
    "mouthPressLeft": {"name": "薄笑い左", "panel": MORPH_LIP},
    "mouthPress": {"name": "薄笑い", "panel": MORPH_LIP, "binds": ["mouthPressRight", "mouthPressLeft"]},
    "mouthSmileRight": {"name": "にやり2右", "panel": MORPH_LIP},
    "mouthSmileLeft": {"name": "にやり2左", "panel": MORPH_LIP},
    "mouthSmile": {"name": "にやり2", "panel": MORPH_LIP, "binds": ["mouthSmileRight", "mouthSmileLeft"]},
    "mouthUpperUpRight": {"name": "にひ右", "panel": MORPH_LIP},
    "mouthUpperUpLeft": {"name": "にひ左", "panel": MORPH_LIP},
    "mouthUpperUp": {"name": "にひ", "panel": MORPH_LIP, "binds": ["mouthUpperUpRight", "mouthDimpleLeft"]},
    "cheekSquintRight": {"name": "にひひ右", "panel": MORPH_LIP},
    "cheekSquintLeft": {"name": "にひひ左", "panel": MORPH_LIP},
    "cheekSquint": {"name": "にひひ", "panel": MORPH_LIP, "binds": ["cheekSquintRight", "cheekSquintLeft"]},
    "mouthFrownRight": {"name": "ちっ右", "panel": MORPH_LIP},
    "mouthFrownLeft": {"name": "ちっ左", "panel": MORPH_LIP},
    "mouthFrown": {"name": "ちっ", "panel": MORPH_LIP, "binds": ["mouthFrownRight", "mouthFrownLeft"]},
    "mouthLowerDownRight": {"name": "むっ右", "panel": MORPH_LIP},
    "mouthLowerDownLeft": {"name": "むっ左", "panel": MORPH_LIP},
    "mouthLowerDown": {"name": "むっ", "panel": MORPH_LIP, "binds": ["mouthLowerDownRight", "mouthLowerDownLeft"]},
    "mouthStretchRight": {"name": "ぎりっ右", "panel": MORPH_LIP},
    "mouthStretchLeft": {"name": "ぎりっ左", "panel": MORPH_LIP},
    "mouthStretch": {"name": "ぎりっ", "panel": MORPH_LIP, "binds": ["mouthStretchRight", "mouthStretchLeft"]},
    "tongueOut": {"name": "べー", "panel": MORPH_LIP},
    "_mouthFunnel+SharpenLips": {"name": "うほっ", "panel": MORPH_LIP},
    "_mouthPress+CatMouth": {"name": "ω口", "panel": MORPH_LIP},
    "_mouthPress+CatMouth-ex": {"name": "ω口2", "panel": MORPH_LIP},
    "_mouthPress+DuckMouth": {"name": "ω口3", "panel": MORPH_LIP},
    "cheekPuff_R": {"name": "ぷくー右", "panel": MORPH_LIP, "split": "cheekPuff"},
    "cheekPuff_L": {"name": "ぷくー左", "panel": MORPH_LIP, "split": "cheekPuff"},
    "cheekPuff": {"name": "ぷくー", "panel": MORPH_LIP},
    "Fcl_MTH_SkinFung_L": {"name": "肌牙左", "panel": MORPH_LIP},
    "Fcl_MTH_SkinFung_R": {"name": "肌牙右", "panel": MORPH_LIP},
    "Fcl_MTH_SkinFung": {"name": "肌牙", "panel": MORPH_LIP},
    "Fcl_HA_Fung1": {"name": "牙", "panel": MORPH_LIP},
    "Fcl_HA_Fung1_Up_R": {"name": "牙上右", "panel": MORPH_LIP, "split": "Fcl_HA_Fung1_Up"},
    "Fcl_HA_Fung1_Up_L": {"name": "牙上左", "panel": MORPH_LIP, "split": "Fcl_HA_Fung1_Up"},
    "Fcl_HA_Fung1_Up": {"name": "牙上", "panel": MORPH_LIP},
    "Fcl_HA_Fung1_Low_R": {"name": "牙下右", "panel": MORPH_LIP, "split": "Fcl_HA_Fung1_Low"},
    "Fcl_HA_Fung1_Low_L": {"name": "牙下左", "panel": MORPH_LIP, "split": "Fcl_HA_Fung1_Low"},
    "Fcl_HA_Fung1_Low": {"name": "牙下", "panel": MORPH_LIP},
    "Fcl_HA_Fung2_Up": {"name": "ギザ歯上", "panel": MORPH_LIP},
    "Fcl_HA_Fung2_Low": {"name": "ギザ歯下", "panel": MORPH_LIP},
    "Fcl_HA_Fung2": {"name": "ギザ歯", "panel": MORPH_LIP},
    "Fcl_HA_Fung3_Up": {"name": "真ん中牙上", "panel": MORPH_LIP},
    "Fcl_HA_Fung3_Low": {"name": "真ん中牙下", "panel": MORPH_LIP},
    "Fcl_HA_Fung3": {"name": "真ん中牙", "panel": MORPH_LIP},
    "Fcl_HA_Hide": {"name": "歯隠", "panel": MORPH_LIP},
    "Fcl_HA_Short_Up": {"name": "歯短上", "panel": MORPH_LIP},
    "Fcl_HA_Short_Low": {"name": "歯短下", "panel": MORPH_LIP},
    "Fcl_HA_Short": {"name": "歯短", "panel": MORPH_LIP},
    "Cheek_Dye": {"name": "照れ", "panel": MORPH_OTHER, "material": "cheek_dye"},
    "Fcl_ALL_Neutral": {"name": "ニュートラル", "panel": MORPH_OTHER},
    "Fcl_ALL_Angry": {"name": "怒", "panel": MORPH_OTHER},
    "Fcl_ALL_Fun": {"name": "楽", "panel": MORPH_OTHER},
    "Fcl_ALL_Joy": {"name": "喜", "panel": MORPH_OTHER},
    "Fcl_ALL_Sorrow": {"name": "哀", "panel": MORPH_OTHER},
    "Fcl_ALL_Surprised": {"name": "驚", "panel": MORPH_OTHER},
    "Edge_Off": {"name": "エッジOFF", "panel": MORPH_OTHER, "edge": True},
}
