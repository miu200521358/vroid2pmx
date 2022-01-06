# -*- coding: utf-8 -*-
#
import logging
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
# import _pickle as cPickle

from module.MOptions import MExportOptions
from mmd.PmxData import PmxModel, Vertex, Material, Bone, Morph, DisplaySlot, RigidBody, Joint, Bdef1, Bdef2, Bdef4, Sdef, RigidBodyParam, IkLink, Ik, BoneMorphData # noqa
from mmd.PmxData import Bdef1, Bdef2, Bdef4, VertexMorphOffset, GroupMorphData # noqa
from mmd.PmxWriter import PmxWriter
from mmd.VmdData import VmdMotion, VmdBoneFrame, VmdCameraFrame, VmdInfoIk, VmdLightFrame, VmdMorphFrame, VmdShadowFrame, VmdShowIkFrame # noqa
from module.MMath import MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from utils import MServiceUtils
from utils.MLogger import MLogger # noqa
from utils.MException import SizingException, MKilledException

logger = MLogger(__name__, level=1)

MIME_TYPE = {
    'image/png': 'png',
    'image/jpeg': 'jpg',
    'image/ktx': 'ktx',
    'image/ktx2': 'ktx2',
    'image/webp': 'webp',
    'image/vnd-ms.dds': 'dds',
    'audio/wav': 'wav'
}

# MMDにおける1cm＝0.125(ミクセル)、1m＝12.5
MIKU_METER = 12.5


class VroidExportService():
    def __init__(self, options: MExportOptions):
        self.options = options
        self.offset = 0
        self.buffer = None

    def execute(self):
        logging.basicConfig(level=self.options.logging_level, format="%(message)s [%(module_name)s]")

        try:
            service_data_txt = f"{logger.transtext('Vroid2Pmx処理実行')}\n------------------------\n{logger.transtext('exeバージョン')}: {self.options.version_name}\n"
            service_data_txt = f"{service_data_txt}　{logger.transtext('元モデル')}: {os.path.basename(self.options.pmx_model.path)}\n"

            logger.info(service_data_txt, translate=False, decoration=MLogger.DECORATION_BOX)

            model = self.vroid2pmx()
            if not model:
                return False

            # 最後に出力
            logger.info("PMX出力開始", decoration=MLogger.DECORATION_LINE)

            os.makedirs(os.path.dirname(self.options.output_path), exist_ok=True)
            PmxWriter().write(model, self.options.output_path)

            logger.info("出力終了: %s", os.path.basename(self.options.output_path), decoration=MLogger.DECORATION_BOX, title="成功")

            return True
        except MKilledException:
            return False
        except SizingException as se:
            logger.error("Vroid2Pmx処理が処理できないデータで終了しました。\n\n%s", se.message, decoration=MLogger.DECORATION_BOX)
        except Exception:
            logger.critical("Vroid2Pmx処理が意図せぬエラーで終了しました。\n\n%s", traceback.format_exc(), decoration=MLogger.DECORATION_BOX)
        finally:
            logging.shutdown()

    def vroid2pmx(self):
        try:
            model, tex_dir_path = self.create_model()
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

            model = self.transfer_astance(model)
            if not model:
                return False
            
            bone_vertices = self.create_bone_vertices(model)

            model = self.create_body_physics(model, bone_vertices)
            if not model:
                return False

            return model
        except MKilledException as ke:
            # 終了命令
            raise ke
        except SizingException as se:
            logger.error("Vroid2Pmx処理が処理できないデータで終了しました。\n\n%s", se.message, decoration=MLogger.DECORATION_BOX)
            return se
        except Exception as e:
            import traceback
            logger.critical("Vroid2Pmx処理が意図せぬエラーで終了しました。\n\n%s", traceback.format_exc(), decoration=MLogger.DECORATION_BOX)
            raise e

    def create_body_physics(self, model: PmxModel, bone_vertices: dict):
        for bone_name, bone in model.bones.items():
            if bone.index not in bone_vertices or bone.english_name not in BONE_PAIRS or (bone.english_name in BONE_PAIRS and BONE_PAIRS[bone.english_name]['rigidbodyGroup'] < 0):
                # ボーンに紐付く頂点がないか、人体ボーンで対象外の場合、スルー
                continue

            rigidbody_shape = BONE_PAIRS[bone.english_name]['rigidbodyShape']
            rigidbody_mode = BONE_PAIRS[bone.english_name]['rigidbodyMode']
            collision_group = BONE_PAIRS[bone.english_name]['rigidbodyGroup']
            no_collision_group_list = BONE_PAIRS[bone.english_name]['rigidbodyNoColl']

            no_collision_group = 0
            for nc in range(16):
                if nc not in no_collision_group_list:
                    no_collision_group |= 1 << nc

            # 剛体生成対象の場合のみ作成
            vertex_list = []
            normal_list = []
            for vertex in bone_vertices[bone.index]:
                vertex_list.append(vertex.position.data().tolist())
                normal_list.append(vertex.normal.data().tolist())
            vertex_ary = np.array(vertex_list)
            # 法線の平均値
            mean_normal = np.mean(np.array(vertex_list), axis=0)
            # 最小
            min_vertex = np.min(vertex_ary, axis=0)
            # 最大
            max_vertex = np.max(vertex_ary, axis=0)
            # 中央
            center_vertex = np.median(vertex_ary, axis=0)

            # ボーンの向き先に沿う
            tail_bone = None
            if bone.tail_index > 0:
                tail_bone = [b for b in model.bones.values() if bone.tail_index == b.index][0]
                tail_position = tail_bone.position
            else:
                tail_position = bone.tail_position + bone.position

            # サイズ
            diff_size = np.abs(max_vertex - min_vertex)
            shape_size = MVector3D()
            shape_rotation = MVector3D()
            if rigidbody_shape == 0:
                # 球体
                if "頭" == bone.name:
                    # 頭はエルフ耳がある場合があるので、両目の間隔を使う
                    eye_length = model.bones["右目"].position.distanceToPoint(model.bones["左目"].position) * 2.5
                    center_vertex[0] = bone.position.x()
                    center_vertex[1] = min_vertex[1] + (max_vertex[1] - min_vertex[1]) / 2
                    center_vertex[2] = bone.position.z()
                    shape_size = MVector3D(eye_length, eye_length, eye_length)
                else:
                    # それ以外（胸とか）はそのまま
                    max_size = np.max(diff_size / 2)
                    shape_size = MVector3D(max_size, max_size, max_size)
                    center_vertex = bone.position
            else:
                # カプセルと箱
                axis_vec = tail_position - bone.position
                tail_pos = axis_vec.normalized()
                tail_vec = tail_pos.data()
                
                # 回転量
                to_vec = MVector3D.crossProduct(MVector3D(mean_normal), MVector3D(tail_vec)).normalized()
                if rigidbody_shape == 1:
                    # 箱
                    rot = MQuaternion.rotationTo(MVector3D(0, 1 * np.sign(tail_vec[1]), 0), tail_pos)
                    rot *= MQuaternion.rotationTo(MVector3D(1 * np.sign(tail_vec[0]), 0, 0), to_vec.normalized())
                else:
                    # カプセル
                    rot = MQuaternion.rotationTo(MVector3D(0, 1, 0), tail_pos)
                    if '上' in bone.name or '下' in bone.name:
                        # 体幹は横に倒しておく
                        rot *= MQuaternion.fromEulerAngles(0, 0, 90)
                shape_euler = rot.toEulerAngles()
                shape_rotation = MVector3D(math.radians(shape_euler.x()), math.radians(shape_euler.y()), math.radians(shape_euler.z()))

                # 軸の長さ
                if rigidbody_shape == 1:
                    # 箱
                    if (tail_bone and tail_bone.tail_index == -1):
                        # 末端の場合、頂点の距離感で決める
                        shape_size = MVector3D(diff_size[0] * 0.7, diff_size[1] * 0.7, diff_size[2] * 0.15)
                    else:
                        # 途中の場合、ボーンの距離感で決める
                        shape_size = MVector3D(diff_size[0] * 0.7, (bone.position.y() - tail_position.y()) * 0.7, diff_size[2] * 0.15)
                        center_vertex = bone.position + (tail_position - bone.position) / 2
                else:
                    # カプセル
                    if ('左' in bone.name or '右' in bone.name) and ('腕' in bone.name or 'ひじ' in bone.name or '手首' in bone.name):
                        # 腕の場合 / 半径：Y, 高さ：X
                        shape_size = MVector3D(diff_size[1] * 0.26, abs(axis_vec.x() * 1), diff_size[2])
                    elif '上' in bone.name or '下' in bone.name:
                        # 体幹の場合 / 半径：X, 高さ：Y
                        shape_size = MVector3D(diff_size[0] * 0.4, abs(axis_vec.y() * 0.5), diff_size[2])
                    elif '首' == bone.name:
                        # 首の場合 / 半径：X, 高さ：Y
                        shape_size = MVector3D(diff_size[0] * 0.2, abs(axis_vec.y() * 0.8), diff_size[2])
                    else:
                        # 足の場合 / 半径：X, 高さ：Y
                        shape_size = MVector3D(diff_size[0] * 0.55, abs(axis_vec.y() * 1), diff_size[2])

                    center_vertex = bone.position + (tail_position - bone.position) / 2

            logger.debug("bone: %s, min: %s, max: %s, center: %s, size: %s", bone.name, min_vertex, max_vertex, center_vertex, shape_size.to_log())
            rigidbody = RigidBody(bone.name, bone.english_name, bone.index, collision_group, no_collision_group, \
                                  rigidbody_shape, shape_size, MVector3D(center_vertex), shape_rotation, 1, 0.5, 0.5, 0, 0, rigidbody_mode)
            rigidbody.index = len(model.rigidbodies)
            model.rigidbodies[rigidbody.name] = rigidbody

        logger.info("-- 身体剛体設定終了")

        return model
    
    def create_bone_vertices(self, model: PmxModel):
        bone_vertices = {}
        for vertex in model.vertex_dict.values():
            for bone_idx in vertex.deform.get_idx_list(0.3):
                if bone_idx not in bone_vertices:
                    bone_vertices[bone_idx] = []
                bone = model.bones[model.bone_indexes[bone_idx]]
                bone_vertices[bone_idx].append(vertex)

                if "捩" in bone.name:
                    # 捩りは親に入れる
                    if bone.parent_index not in bone_vertices:
                        bone_vertices[bone.parent_index] = []
                    bone_vertices[bone.parent_index].append(vertex)
                elif bone.getExternalRotationFlag():
                    # 回転付与の場合、付与親に入れる
                    if bone.effect_index not in bone_vertices:
                        bone_vertices[bone.effect_index] = []
                    bone_vertices[bone.effect_index].append(vertex)

        return bone_vertices
    
    def transfer_astance(self, model: PmxModel):
        # 各頂点
        all_vertex_relative_poses = {}
        for vertex in model.vertex_dict.values():
            if type(vertex.deform) is Bdef1:
                all_vertex_relative_poses[vertex.index] = [vertex.position - model.bones[model.bone_indexes[vertex.deform.index0]].position]
            elif type(vertex.deform) is Bdef2:
                all_vertex_relative_poses[vertex.index] = [vertex.position - model.bones[model.bone_indexes[vertex.deform.index0]].position, \
                                                           vertex.position - model.bones[model.bone_indexes[vertex.deform.index1]].position]
            elif type(vertex.deform) is Bdef4:
                all_vertex_relative_poses[vertex.index] = [vertex.position - model.bones[model.bone_indexes[vertex.deform.index0]].position, \
                                                           vertex.position - model.bones[model.bone_indexes[vertex.deform.index1]].position, \
                                                           vertex.position - model.bones[model.bone_indexes[vertex.deform.index2]].position, \
                                                           vertex.position - model.bones[model.bone_indexes[vertex.deform.index3]].position]

        for direction, astance_qq in [("右", MQuaternion.fromEulerAngles(0, 0, 35)), ("左", MQuaternion.fromEulerAngles(0, 0, -35))]:
            trans_bone_vecs = {}
            trans_bone_mats = {}
            trans_vertex_vecs = {}
            trans_normal_vecs = {}

            trans_bone_mats["全ての親"] = MMatrix4x4()
            trans_bone_mats["全ての親"].setToIdentity()

            arm_bone_name = f'{direction}腕'
            bone_names = ['頭', f'{direction}親指先', f'{direction}人指先', f'{direction}中指先', f'{direction}薬指先', f'{direction}小指先', \
                          f'{direction}胸先', f'{direction}腕捩1', f'{direction}腕捩2', f'{direction}腕捩3', f'{direction}手捩1', f'{direction}手捩2', f'{direction}手捩3']
            
            for bname in model.bones.keys():
                if '装飾_' in bname:
                    bone_names.append(bname)

            for end_bone_name in bone_names:
                bone_links = model.create_link_2_top_one(end_bone_name, is_defined=False).to_links('上半身')
                if len(bone_links.all().keys()) == 0:
                    continue

                trans_vs = MServiceUtils.calc_relative_position(model, bone_links, VmdMotion(), 0)

                mat = MMatrix4x4()
                mat.setToIdentity()
                for vi, (bone_name, trans_v) in enumerate(zip(bone_links.all().keys(), trans_vs)):
                    mat.translate(trans_v)
                    if bone_name == arm_bone_name:
                        # 腕だけ回転させる
                        mat.rotate(astance_qq)
                    
                    if bone_name not in trans_bone_vecs:
                        trans_bone_vecs[bone_name] = mat * MVector3D()
                        trans_bone_mats[bone_name] = mat.copy()
            
            for bone_name, bone_vec in trans_bone_vecs.items():
                model.bones[bone_name].position = bone_vec

            local_y_vector = MVector3D(0, -1, 0)
            # local_z_vector = MVector3D(0, 0, -1)
            for bone_name, bone_mat in trans_bone_mats.items():
                bone = model.bones[bone_name]
                arm_bone_name = f'{direction}腕'
                elbow_bone_name = f'{direction}ひじ'
                wrist_bone_name = f'{direction}手首'
                finger_bone_name = f'{direction}中指１'
                
                # ローカル軸
                if bone.name in ['右肩', '左肩'] and arm_bone_name in model.bones:
                    bone.local_x_vector = (model.bones[arm_bone_name].position - model.bones[bone.name].position).normalized()
                    bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()
                if bone.name in ['右腕', '左腕'] and elbow_bone_name in model.bones:
                    bone.local_x_vector = (model.bones[elbow_bone_name].position - model.bones[bone.name].position).normalized()
                    bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()
                if bone.name in ['右ひじ', '左ひじ'] and wrist_bone_name in model.bones:
                    # ローカルYで曲げる
                    bone.local_x_vector = (model.bones[wrist_bone_name].position - model.bones[bone.name].position).normalized()
                    bone.local_z_vector = MVector3D.crossProduct(local_y_vector, bone.local_x_vector).normalized()
                if bone.name in ['右手首', '左手首'] and finger_bone_name in model.bones:
                    bone.local_x_vector = (model.bones[finger_bone_name].position - model.bones[bone.name].position).normalized()
                    bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()
                # 捩り
                if bone.name in ['右腕捩', '左腕捩'] and arm_bone_name in model.bones and elbow_bone_name in model.bones:
                    bone.fixed_axis = (model.bones[elbow_bone_name].position - model.bones[arm_bone_name].position).normalized()
                    bone.local_x_vector = (model.bones[elbow_bone_name].position - model.bones[arm_bone_name].position).normalized()
                    bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()
                if bone.name in ['右手捩', '左手捩'] and elbow_bone_name in model.bones and wrist_bone_name in model.bones:
                    bone.fixed_axis = (model.bones[wrist_bone_name].position - model.bones[elbow_bone_name].position).normalized()
                    bone.local_x_vector = (model.bones[wrist_bone_name].position - model.bones[elbow_bone_name].position).normalized()
                    bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()
                # 指
                if bone.english_name in BONE_PAIRS and BONE_PAIRS[bone.english_name]['display'] and '指' in BONE_PAIRS[bone.english_name]['display']:
                    bone.local_x_vector = (model.bones[model.bone_indexes[bone.tail_index]].position - model.bones[model.bone_indexes[bone.parent_index]].position).normalized()    # noqa
                    bone.local_z_vector = MVector3D.crossProduct(bone.local_x_vector, local_y_vector).normalized()

            for vertex_idx, vertex_relative_poses in all_vertex_relative_poses.items():
                if vertex_idx not in trans_vertex_vecs:
                    vertex = model.vertex_dict[vertex_idx]
                    if type(vertex.deform) is Bdef1 and model.bone_indexes[vertex.deform.index0] in trans_bone_mats:
                        trans_vertex_vecs[vertex.index] = trans_bone_mats[model.bone_indexes[vertex.deform.index0]] * vertex_relative_poses[0]
                        trans_normal_vecs[vertex.index] = self.calc_normal(trans_bone_mats[model.bone_indexes[vertex.deform.index0]], vertex.normal)
                    elif type(vertex.deform) is Bdef2 and (model.bone_indexes[vertex.deform.index0] in trans_bone_mats and model.bone_indexes[vertex.deform.index1] in trans_bone_mats):
                        v0_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index0]] * vertex_relative_poses[0]
                        v1_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index1]] * vertex_relative_poses[1]
                        trans_vertex_vecs[vertex.index] = (v0_vec * vertex.deform.weight0) + (v1_vec * (1 - vertex.deform.weight0))

                        v0_normal = self.calc_normal(trans_bone_mats[model.bone_indexes[vertex.deform.index0]], vertex.normal)
                        v1_normal = self.calc_normal(trans_bone_mats[model.bone_indexes[vertex.deform.index1]], vertex.normal)
                        trans_normal_vecs[vertex.index] = (v0_normal * vertex.deform.weight0) + (v1_normal * (1 - vertex.deform.weight0))
                    elif type(vertex.deform) is Bdef4 and (model.bone_indexes[vertex.deform.index0] in trans_bone_mats and model.bone_indexes[vertex.deform.index1] in trans_bone_mats \
                                                        and model.bone_indexes[vertex.deform.index2] in trans_bone_mats and model.bone_indexes[vertex.deform.index3] in trans_bone_mats):
                        v0_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index0]] * vertex_relative_poses[0]
                        v1_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index1]] * vertex_relative_poses[1]
                        v2_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index2]] * vertex_relative_poses[2]
                        v3_vec = trans_bone_mats[model.bone_indexes[vertex.deform.index3]] * vertex_relative_poses[3]
                        trans_vertex_vecs[vertex.index] = (v0_vec * vertex.deform.weight0) + (v1_vec * vertex.deform.weight1) + (v2_vec * vertex.deform.weight2) + (v3_vec * vertex.deform.weight3)

                        v0_normal = self.calc_normal(trans_bone_mats[model.bone_indexes[vertex.deform.index0]], vertex.normal)
                        v1_normal = self.calc_normal(trans_bone_mats[model.bone_indexes[vertex.deform.index1]], vertex.normal)
                        v2_normal = self.calc_normal(trans_bone_mats[model.bone_indexes[vertex.deform.index2]], vertex.normal)
                        v3_normal = self.calc_normal(trans_bone_mats[model.bone_indexes[vertex.deform.index3]], vertex.normal)
                        trans_normal_vecs[vertex.index] = (v0_normal * vertex.deform.weight0) + (v1_normal * vertex.deform.weight1) + (v2_normal * vertex.deform.weight2) + (v3_normal * vertex.deform.weight3)     # noqa

            for (vertex_idx, vertex_vec), (_, vertex_normal) in zip(trans_vertex_vecs.items(), trans_normal_vecs.items()):
                model.vertex_dict[vertex_idx].position = vertex_vec
                model.vertex_dict[vertex_idx].normal = vertex_normal.normalized()
                        
        logger.info("-- Aスタンス調整終了")

        return model
    
    def calc_normal(self, bone_mat: MMatrix4x4, normal: MVector3D):
        # ボーン行列の3x3行列
        bone_invert_mat = bone_mat.data()[:3, :3]

        return MVector3D(np.sum(normal.data() * bone_invert_mat, axis=1)).normalized()
    
    def convert_morph(self, model: PmxModel):
        # グループモーフ定義
        if "extensions" not in model.json_data or "VRM" not in model.json_data["extensions"] \
                or "blendShapeMaster" not in model.json_data["extensions"]["VRM"] or "blendShapeGroups" not in model.json_data["extensions"]["VRM"]["blendShapeMaster"]:
            return model

        # 定義済みグループモーフ
        for sidx, shape in enumerate(model.json_data["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]):
            if len(shape["binds"]) == 0:
                continue

            morph_name = shape["name"]
            morph_panel = 4
            if shape["name"] in MORPH_PAIRS:
                morph_name = MORPH_PAIRS[shape["name"]]["name"]
                morph_panel = MORPH_PAIRS[shape["name"]]["panel"]
            morph = Morph(morph_name, shape["name"], morph_panel, 0)
            morph.index = len(model.org_morphs)
            
            if shape["name"] in MORPH_PAIRS and "binds" in MORPH_PAIRS[shape["name"]]:
                for bind in MORPH_PAIRS[shape["name"]]["binds"]:
                    morph.offsets.append(GroupMorphData(model.org_morphs[bind].index, 1))
            else:
                for bind in shape["binds"]:
                    morph.offsets.append(GroupMorphData(bind["index"], bind["weight"] / 100))
            model.org_morphs[morph_name] = morph
            if morph_name not in DEFINED_MORPH_NAMES:
                model.display_slots["表情"].references.append((1, morph.index))

        # 自前グループモーフ
        for sidx, (morph_name, morph_pair) in enumerate(MORPH_PAIRS.items()):
            if "binds" in morph_pair:
                # 統合グループモーフ（ある場合だけ）
                morph = Morph(morph_pair["name"], morph_name, morph_pair["panel"], 0)
                morph.index = len(model.org_morphs)
                for bind_name in morph_pair["binds"]:
                    if bind_name in model.org_morphs:
                        bind_morph = model.org_morphs[bind_name]
                        morph.offsets.append(GroupMorphData(bind_morph.index, 1))
                if len(morph.offsets) > 0:
                    model.org_morphs[morph_pair["name"]] = morph
                    model.display_slots["表情"].references.append((1, morph.index))
            elif "split" in morph_pair:
                if morph_pair["split"] in model.org_morphs:
                    # 元のモーフを左右に分割する
                    org_morph = model.org_morphs[morph_pair["split"]]
                    target_offset = []
                    if org_morph.morph_type == 1:
                        for offset in org_morph.offsets:
                            vertex = model.vertex_dict[offset.vertex_index]
                            if ("_R" == morph_name[-2:] and vertex.position.x() < 0) or ("_L" == morph_name[-2:] and vertex.position.x() > 0):
                                if morph_pair["panel"] == MORPH_LIP:
                                    # リップは中央にいくに従ってオフセットを弱める(最大値は0.7)
                                    ratio = 1 if abs(vertex.position.x()) >= 0.2 else calc_ratio(abs(vertex.position.x()), 0, 0.2, 0, 0.7)
                                    target_offset.append(VertexMorphOffset(offset.vertex_index, offset.position_offset * ratio))
                                else:
                                    target_offset.append(VertexMorphOffset(offset.vertex_index, offset.position_offset.copy()))
                    if target_offset:
                        morph = Morph(morph_pair["name"], morph_name, morph_pair["panel"], 1)
                        morph.index = len(model.org_morphs)
                        morph.offsets = target_offset

                        model.org_morphs[morph_pair["name"]] = morph
                        model.display_slots["表情"].references.append((1, morph.index))
            else:
                if morph_name in model.org_morphs:
                    if morph_pair["panel"] == MORPH_LIP and morph_pair["ratio"] != 1:
                        # リップで倍率縮小
                        # 元モーフの名前を変更
                        model.org_morphs[morph_name].name = f'{morph_pair["name"]}(1)'

                        # リップモーフは0.7倍にするため、グループモーフにする
                        morph = Morph(morph_pair["name"], morph_pair["name"], morph_pair["panel"], 0)
                        morph.index = len(model.org_morphs)
                        morph.offsets.append(GroupMorphData(model.org_morphs[morph_name].index, morph_pair["ratio"]))

                        model.org_morphs[morph_pair["name"]] = morph
                    else:
                        # それ以外は名前のみ置換
                        morph = model.org_morphs[morph_name]
                        morph.name = morph_pair["name"]
                        morph.panel = morph_pair["panel"]
                    
                    model.display_slots["表情"].references.append((1, morph.index))

        logger.info('-- グループモーフデータ解析')

        return model

    def reconvert_bone(self, model: PmxModel):
        # 指先端の位置を計算して配置
        finger_dict = {'左親指２': {'vertices': [], 'direction': -1, 'edge_name': '左親指先'}, '左人指３': {'vertices': [], 'direction': -1, 'edge_name': '左人指先'}, \
                       '左中指３': {'vertices': [], 'direction': -1, 'edge_name': '左中指先'}, '左薬指３': {'vertices': [], 'direction': -1, 'edge_name': '左薬指先'}, \
                       '左小指３': {'vertices': [], 'direction': -1, 'edge_name': '左小指先'}, '右親指２': {'vertices': [], 'direction': 1, 'edge_name': '右親指先'}, \
                       '右人指３': {'vertices': [], 'direction': 1, 'edge_name': '右人指先'}, '右中指３': {'vertices': [], 'direction': 1, 'edge_name': '右中指先'}, \
                       '右薬指３': {'vertices': [], 'direction': 1, 'edge_name': '右薬指先'}, '右小指３': {'vertices': [], 'direction': 1, 'edge_name': '右小指先'}}
        # つま先の位置を計算して配置
        toe_dict = {'左足先EX': {'vertices': [], 'edge_name': '左つま先', 'ik_name': '左つま先ＩＫ'}, '右足先EX': {'vertices': [], 'edge_name': '右つま先', 'ik_name': '右つま先ＩＫ'}}
        
        for vertex_idx, vertex in model.vertex_dict.items():
            if type(vertex.deform) is Bdef1:
                # 指先に相当する頂点位置をリスト化
                for finger_name in finger_dict.keys():
                    if model.bones[finger_name].index == vertex.deform.index0:
                        finger_dict[finger_name]['vertices'].append(vertex.position)
                # つま先に相当する頂点位置をリスト化
                for toe_name in toe_dict.keys():
                    if model.bones[toe_name].index == vertex.deform.index0:
                        toe_dict[toe_name]['vertices'].append(vertex.position)
        
        for finger_name, finger_param in finger_dict.items():
            if len(finger_param['vertices']) > 0:
                # 末端頂点の位置を指先ボーンの位置として割り当て
                finger_vertices = sorted(finger_param['vertices'], key=lambda v: v.x() * finger_param['direction'])
                edge_vertex_pos = finger_vertices[0]
                model.bones[finger_param['edge_name']].position = edge_vertex_pos

        for toe_name, toe_param in toe_dict.items():
            if len(toe_param['vertices']) > 0:
                # 末端頂点の位置をつま先ボーンの位置として割り当て
                toe_vertices = sorted(toe_param['vertices'], key=lambda v: v.z())
                edge_vertex_pos = toe_vertices[0].copy()
                # Yは0に固定
                edge_vertex_pos.setY(0)
                model.bones[toe_param['edge_name']].position = edge_vertex_pos
                model.bones[toe_param['ik_name']].position = edge_vertex_pos
        
        for leg_bone_name in ['腰キャンセル左', '腰キャンセル右', '左足', '右足', '左足D', '右足D']:
            if leg_bone_name in model.bones:
                model.bones[leg_bone_name].position.setZ(model.bones[leg_bone_name].position.z() + 0.1)

        for knee_bone_name in ['左ひざ', '右ひざ', '左ひざD', '右ひざD']:
            if knee_bone_name in model.bones:
                model.bones[knee_bone_name].position.setZ(model.bones[knee_bone_name].position.z() - 0.1)
        
        # 体幹を中心に揃える
        for trunk_bone_name in ["全ての親", "センター", "グルーブ", "腰", "下半身", "上半身", "上半身2", "上半身3", "首", "頭", "両目"]:
            model.bones[trunk_bone_name].position.setX(0)

        # 左右ボーンを線対称に揃える
        for left_bone_name, left_bone in model.bones.items():
            right_bone_name = f'右{left_bone_name[1:]}'
            if '左' == left_bone_name[0] and right_bone_name in model.bones:
                right_bone = model.bones[right_bone_name]
                mean_position = MVector3D(np.mean([abs(left_bone.position.x()), abs(right_bone.position.x())]), \
                                          np.mean([left_bone.position.y(), right_bone.position.y()]), np.mean([left_bone.position.z(), right_bone.position.z()]))
                left_bone.position = MVector3D(mean_position.x() * np.sign(left_bone.position.x()), mean_position.y(), mean_position.z())
                right_bone.position = MVector3D(mean_position.x() * np.sign(right_bone.position.x()), mean_position.y(), mean_position.z())

        logger.info("-- ボーンデータ調整終了")

        return model

    def convert_mesh(self, model: PmxModel, bone_name_dict: dict, tex_dir_path: str):
        if 'meshes' not in model.json_data:
            logger.error("変換可能なメッシュ情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
            return None

        vertex_blocks = {}
        vertex_idx = 0

        for midx, mesh in enumerate(model.json_data["meshes"]):
            if "primitives" not in mesh:
                continue
            
            for pidx, primitive in enumerate(mesh["primitives"]):
                if "attributes" not in primitive or "indices" not in primitive or "material" not in primitive or "JOINTS_0" not in primitive["attributes"] or "NORMAL" not in primitive["attributes"] \
                   or "POSITION" not in primitive["attributes"] or "TEXCOORD_0" not in primitive["attributes"] or "WEIGHTS_0" not in primitive["attributes"]:
                    continue
                
                # 頂点ブロック
                vertex_key = f'{primitive["attributes"]["JOINTS_0"]}-{primitive["attributes"]["NORMAL"]}-{primitive["attributes"]["POSITION"]}-{primitive["attributes"]["TEXCOORD_0"]}-{primitive["attributes"]["WEIGHTS_0"]}'  # noqa

                # 頂点データ
                if vertex_key not in vertex_blocks:
                    vertex_blocks[vertex_key] = {'vertices': [], 'start': vertex_idx, 'indices': [], 'materials': []}

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
                        skin_joints = model.json_data["skins"][[s for s in model.json_data["nodes"] if "mesh" in s and s["mesh"] == midx][0]["skin"]]["joints"]
                    except Exception:
                        # 取れない場合はとりあえず空
                        skin_joints = []
                        
                    if "extras" in primitive and "targetNames" in primitive["extras"] and "targets" in primitive:
                        for eidx, (extra, target) in enumerate(zip(primitive["extras"]["targetNames"], primitive["targets"])):
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
                        joint_idxs, weight_values = self.get_deform_index(vertex_idx, model, model_position, joint, skin_joints, weight, bone_name_dict)
                        if len(joint_idxs) > 1:
                            if len(joint_idxs) == 2:
                                # ウェイトが2つの場合、Bdef2
                                deform = Bdef2(joint_idxs[0], joint_idxs[1], weight_values[0])
                            else:
                                # それ以上の場合、Bdef4
                                deform = Bdef4(joint_idxs[0], joint_idxs[1], joint_idxs[2], joint_idxs[3], \
                                               weight_values[0], weight_values[1], weight_values[2], weight_values[3])
                        elif len(joint_idxs) == 1:
                            # ウェイトが1つのみの場合、Bdef1
                            deform = Bdef1(joint_idxs[0])
                        else:
                            # とりあえず除外
                            deform = Bdef1(0)

                        vertex = Vertex(vertex_idx, model_position, (normal * MVector3D(-1, 1, 1)).normalized(), uv, None, deform, 1)

                        model.vertex_dict[vertex_idx] = vertex
                        vertex_blocks[vertex_key]['vertices'].append(vertex_idx)
                        vertex_idx += 1

                    logger.info('-- 頂点データ解析[%s]', vertex_key)
                
                vertex_blocks[vertex_key]['indices'].append(primitive["indices"])
                vertex_blocks[vertex_key]['materials'].append(primitive["material"])

        hair_regexp = r'((N\d+_\d+_Hair_\d+)_HAIR)'
        hair_tex_regexp = r'_(\d+)'

        indices_by_materials = {}
        materials_by_type = {}

        for vertex_key, vertex_dict in vertex_blocks.items():
            start_vidx = vertex_dict['start']
            indices = vertex_dict['indices']
            materials = vertex_dict['materials']

            for index_accessor, material_accessor in zip(indices, materials):
                # 材質データ ---------------
                vrm_material = model.json_data["materials"][material_accessor]
                material_name = vrm_material['name']

                # 材質順番を決める
                material_key = vrm_material["alphaMode"]
                if "EyeIris" in material_name:
                    material_key = "EyeIris"
                if "EyeHighlight" in material_name:
                    material_key = "EyeHighlight"
                if "EyeWhite" in material_name:
                    material_key = "EyeWhite"
                if "Eyelash" in material_name:
                    material_key = "Eyelash"
                if "Eyeline" in material_name:
                    material_key = "Eyeline"
                if "FaceBrow" in material_name:
                    material_key = "FaceBrow"
                if "Lens" in material_name:
                    material_key = "Lens"

                if material_key not in materials_by_type:
                    materials_by_type[material_key] = {}

                if material_name not in materials_by_type[material_key]:
                    # VRMの材質拡張情報
                    material_ext = [m for m in model.json_data["extensions"]["VRM"]["materialProperties"] if m["name"] == material_name][0]
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
                    edge_size = material_ext["floatProperties"]["_OutlineWidth"]

                    # 0番目は空テクスチャなので+1で設定
                    m = re.search(hair_regexp, material_name)
                    if m is not None:
                        # 髪材質の場合、合成
                        hair_img_name = os.path.basename(model.textures[material_ext["textureProperties"]["_MainTex"] + 1])
                        hm = re.search(hair_tex_regexp, hair_img_name)
                        hair_img_number = -1
                        if hm is not None:
                            hair_img_number = int(hm.groups()[0])
                        hair_spe_name = f'_{(hair_img_number + 1):02d}.png'
                        hair_blend_name = f'_{hair_img_number:02d}_blend.png'

                        if os.path.exists(os.path.join(tex_dir_path, hair_img_name)) and os.path.exists(os.path.join(tex_dir_path, hair_spe_name)):
                            # スペキュラファイルがある場合
                            hair_img = Image.open(os.path.join(tex_dir_path, hair_img_name)).convert('RGBA')
                            hair_ary = np.array(hair_img)

                            spe_img = Image.open(os.path.join(tex_dir_path, hair_spe_name)).convert('RGBA')
                            spe_ary = np.array(spe_img)

                            # 拡散色の画像
                            diffuse_ary = np.array(material_ext["vectorProperties"]["_Color"])
                            diffuse_img = Image.fromarray(np.tile(diffuse_ary * 255, (hair_ary.shape[0], hair_ary.shape[1], 1)).astype(np.uint8), mode='RGBA')
                            hair_diffuse_img = ImageChops.multiply(hair_img, diffuse_img)

                            # 反射色の画像
                            if "emissiveFactor" in vrm_material:
                                emissive_ary = np.array(vrm_material["emissiveFactor"])
                                emissive_ary = np.append(emissive_ary, 1)
                            else:
                                # なかった場合には仮に明るめの色を入れておく
                                logger.warning("髪の反射色がVrmデータになかったため、仮に白色を差し込みます", decoration=MLogger.DECORATION_BOX)
                                emissive_ary = np.array([0.9, 0.9, 0.9, 1])
                            emissive_img = Image.fromarray(np.tile(emissive_ary * 255, (spe_ary.shape[0], spe_ary.shape[1], 1)).astype(np.uint8), mode='RGBA')
                            # 乗算
                            hair_emissive_img = ImageChops.multiply(spe_img, emissive_img)
                            # スクリーン
                            dest_img = ImageChops.screen(hair_diffuse_img, hair_emissive_img)
                            dest_img.save(os.path.join(tex_dir_path, hair_blend_name))

                            model.textures.append(os.path.join("tex", hair_blend_name))
                            texture_index = len(model.textures) - 1

                            # # 拡散色と環境色は固定
                            # diffuse_color = MVector3D(1, 1, 1)
                            # specular_color = MVector3D()
                            # ambient_color = diffuse_color * 0.5
                        else:
                            # スペキュラがない場合、ないし反映させない場合、そのまま設定
                            texture_index = material_ext["textureProperties"]["_MainTex"] + 1
                    else:
                        # そのまま出力
                        texture_index = material_ext["textureProperties"]["_MainTex"] + 1
                    
                    sphere_texture_index = 0
                    sphere_mode = 0
                    if "_SphereAdd" in material_ext["textureProperties"]:
                        sphere_texture_index = material_ext["textureProperties"]["_SphereAdd"] + 1
                        # 加算スフィア
                        sphere_mode = 2

                    if "vectorProperties" in material_ext and "_ShadeColor" in material_ext["vectorProperties"]:
                        toon_sharing_flag = 0
                        if material_ext["textureProperties"]["_MainTex"] < len(model.json_data["images"]):
                            toon_img_name = f'{model.json_data["images"][material_ext["textureProperties"]["_MainTex"]]["name"]}_Toon.bmp'
                        else:
                            toon_img_name = f'{material_name}_Toon.bmp'
                        
                        toon_light_ary = np.tile(np.array([255, 255, 255, 255]), (24, 32, 1))
                        toon_shadow_ary = np.tile(np.array(material_ext["vectorProperties"]["_ShadeColor"]) * 255, (8, 32, 1))
                        toon_ary = np.concatenate((toon_light_ary, toon_shadow_ary), axis=0)
                        toon_img = Image.fromarray(toon_ary.astype(np.uint8))

                        toon_img.save(os.path.join(tex_dir_path, toon_img_name))
                        model.textures.append(os.path.join("tex", toon_img_name))
                        # 最後に追加したテクスチャをINDEXとして設定
                        toon_texture_index = len(model.textures) - 1
                    else:
                        toon_sharing_flag = 1
                        toon_texture_index = 1

                    material = Material(material_name, material_name, diffuse_color, alpha, specular_factor, specular_color, \
                                        ambient_color, flag, edge_color, edge_size, texture_index, sphere_texture_index, sphere_mode, toon_sharing_flag, \
                                        toon_texture_index, "", 0)
                    materials_by_type[material_key][material.name] = material
                    indices_by_materials[material.name] = {}
                else:
                    material = materials_by_type[material_key][material_name]

                # 面データ ---------------
                indices = self.read_from_accessor(model, index_accessor)
                indices_by_materials[material.name][index_accessor] = (np.array(indices) + start_vidx).tolist()
                material.vertex_count += len(indices)

                logger.info('-- 面・材質データ解析[%s-%s]', index_accessor, material_accessor)
        
        # 材質を不透明(OPAQUE)→透明順(BLEND)に並べ替て設定
        index_idx = 0
        for material_type in ["OPAQUE", "MASK", "BLEND", "FaceBrow", "Eyeline", "Eyelash", "EyeWhite", "EyeIris", "EyeHighlight", "Lens"]:
            if material_type in materials_by_type:
                for material_name, material in materials_by_type[material_type].items():
                    model.materials[material.name] = material
                    model.material_vertices[material.name] = []
                    for index_accessor, indices in indices_by_materials[material.name].items():
                        for v0_idx, v1_idx, v2_idx in zip(indices[:-2:3], indices[1:-1:3], indices[2::3]):
                            # 面の貼り方がPMXは逆
                            model.indices[index_idx] = [v2_idx, v1_idx, v0_idx]
                            index_idx += 1

                            if v0_idx not in model.material_vertices[material.name]:
                                model.material_vertices[material.name].append(v0_idx)

                            if v1_idx not in model.material_vertices[material.name]:
                                model.material_vertices[material.name].append(v1_idx)

                            if v2_idx not in model.material_vertices[material.name]:
                                model.material_vertices[material.name].append(v2_idx)

        logger.info("-- 頂点・面・材質データ解析終了")

        return model
    
    def get_deform_index(self, vertex_idx: int, model: PmxModel, vertex_pos: MVector3D, joint: MVector4D, skin_joints: list, node_weight: list, bone_name_dict: dict):
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
                if bone_param['node_index'] == skin_joints[jidx]:
                    dest_joint_list.append(model.bones[bone_param['name']].index)
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
            for dest_bone_name, src_bone_name in [(f'{direction}足', f'{direction}足D'), (f'{direction}ひざ', f'{direction}ひざD'), \
                                                  (f'{direction}足首', f'{direction}足首D'), (f'{direction}つま先', f'{direction}足先EX')]:
                dest_joints = np.where(dest_joints == model.bones[dest_bone_name].index, model.bones[src_bone_name].index, dest_joints)

            for base_from_name, base_to_name, base_twist_name in [('腕', 'ひじ', '腕捩'), ('ひじ', '手首', '手捩')]:
                dest_arm_bone_name = f'{direction}{base_from_name}'
                dest_elbow_bone_name = f'{direction}{base_to_name}'
                dest_arm_twist1_bone_name = f'{direction}{base_twist_name}1'
                dest_arm_twist2_bone_name = f'{direction}{base_twist_name}2'
                dest_arm_twist3_bone_name = f'{direction}{base_twist_name}3'

                arm_elbow_distance = -1
                vector_arm_distance = 1

                # 腕捩に分散する
                if model.bones[dest_arm_bone_name].index in dest_joints or model.bones[dest_arm_twist1_bone_name].index in dest_joints \
                   or model.bones[dest_arm_twist2_bone_name].index in dest_joints or model.bones[dest_arm_twist3_bone_name].index in dest_joints:
                    # 腕に割り当てられているウェイトの場合
                    arm_elbow_distance = model.bones[dest_elbow_bone_name].position.x() - model.bones[dest_arm_bone_name].position.x()
                    vector_arm_distance = vertex_pos.x() - model.bones[dest_arm_bone_name].position.x()
                    twist_list = [(dest_arm_twist1_bone_name, dest_arm_bone_name), \
                                  (dest_arm_twist2_bone_name, dest_arm_twist1_bone_name), \
                                  (dest_arm_twist3_bone_name, dest_arm_twist2_bone_name)]

                if np.sign(arm_elbow_distance) == np.sign(vector_arm_distance):
                    for dest_to_bone_name, dest_from_bone_name in twist_list:
                        # 腕からひじの間の頂点の場合
                        twist_distance = model.bones[dest_to_bone_name].position.x() - model.bones[dest_from_bone_name].position.x()
                        vector_distance = vertex_pos.x() - model.bones[dest_from_bone_name].position.x()
                        if np.sign(twist_distance) == np.sign(vector_distance):
                            # 腕から腕捩1の間にある頂点の場合
                            arm_twist_factor = vector_distance / twist_distance
                            # 腕が割り当てられているウェイトINDEX
                            arm_twist_weight_joints = np.where(dest_joints == model.bones[dest_from_bone_name].index)[0]
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

                                    logger.test("[%s] from: %s, to: %s, factor: %s, dest_joints: %s, org_weights: %s", \
                                                vertex_idx, dest_from_bone_name, dest_to_bone_name, arm_twist_factor, dest_joints, org_weights)

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
        if 'nodes' not in model.json_data:
            logger.error("変換可能なボーン情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
            return None, None

        # 表示枠 ------------------------
        model.display_slots["全ての親"] = DisplaySlot("Root", "Root", 1, 1)
        model.display_slots["全ての親"].references.append((0, 0))

        # モーフの表示枠
        model.display_slots["表情"] = DisplaySlot("表情", "Exp", 1, 1)

        node_dict = {}
        node_name_dict = {}
        for nidx, node in enumerate(model.json_data['nodes']):
            if 'translation' not in node:
                continue
            
            node = model.json_data['nodes'][nidx]
            logger.debug(f'[{nidx:03d}] node: {node}')

            node_name = node['name']
            
            # 位置
            position = MVector3D(*node['translation']) * MIKU_METER * MVector3D(-1, 1, 1)

            children = node['children'] if 'children' in node else []

            node_dict[nidx] = {'name': node_name, 'relative_position': position, 'position': position, 'parent': -1, 'children': children}
            node_name_dict[node_name] = nidx

        # 親子関係設定
        for nidx, node_param in node_dict.items():
            for midx, parent_node_param in node_dict.items():
                if nidx in parent_node_param['children']:
                    node_dict[nidx]['parent'] = midx
        
        # 絶対位置計算
        for nidx, node_param in node_dict.items():
            node_dict[nidx]['position'] = self.calc_bone_position(model, node_dict, node_param)
        
        # まずは人体ボーン
        bone_name_dict = {}
        for node_name, bone_param in BONE_PAIRS.items():
            parent_name = BONE_PAIRS[bone_param['parent']]['name'] if bone_param['parent'] else None
            parent_index = model.bones[parent_name].index if parent_name else -1

            node_index = -1
            position = MVector3D()
            bone = Bone(bone_param['name'], node_name, position, parent_index, 0, bone_param['flag'])
            if parent_index >= 0:
                if node_name in node_name_dict:
                    node_index = node_name_dict[node_name]
                    position = node_dict[node_name_dict[node_name]]['position'].copy()
                elif node_name == 'Center':
                    position = node_dict[node_name_dict['J_Bip_C_Hips']]['position'] * 0.7
                elif node_name == 'Groove':
                    position = node_dict[node_name_dict['J_Bip_C_Hips']]['position'] * 0.8
                elif node_name == 'J_Bip_C_Spine2':
                    position = node_dict[node_name_dict['J_Bip_C_Spine']]['position'].copy()
                elif node_name == 'J_Adj_FaceEye':
                    position = node_dict[node_name_dict['J_Adj_L_FaceEye']]['position'] + \
                                ((node_dict[node_name_dict['J_Adj_R_FaceEye']]['position'] - node_dict[node_name_dict['J_Adj_L_FaceEye']]['position']) * 0.5)   # noqa
                elif 'shoulderP_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Shoulder']]['position'].copy()
                elif 'shoulderC_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_UpperArm']]['position'].copy()
                    bone.effect_index = bone_name_dict[f'shoulderP_{node_name[-1]}']['index']
                    bone.effect_factor = -1
                elif 'arm_twist_' in node_name:
                    factor = 0.25 if node_name[-2] == '1' else 0.75 if node_name[-2] == '3' else 0.5
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_UpperArm']]['position'] + \
                                ((node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_LowerArm']]['position'] - node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_UpperArm']]['position']) * factor)   # noqa
                    if node_name[-2] in ['1', '2', '3']:
                        bone.effect_index = bone_name_dict[f'arm_twist_{node_name[-1]}']['index']
                        bone.effect_factor = factor
                elif 'wrist_twist_' in node_name:
                    factor = 0.25 if node_name[-2] == '1' else 0.75 if node_name[-2] == '3' else 0.5
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_LowerArm']]['position'] + \
                                ((node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Hand']]['position'] - node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_LowerArm']]['position']) * factor)   # noqa
                    if node_name[-2] in ['1', '2', '3']:
                        bone.effect_index = bone_name_dict[f'wrist_twist_{node_name[-1]}']['index']
                        bone.effect_factor = factor
                elif 'waistCancel_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_UpperLeg']]['position'].copy()
                elif 'leg_IK_Parent_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Foot']]['position'].copy()
                    position.setY(0)
                elif 'leg_IK_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Foot']]['position'].copy()
                elif 'toe_IK_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_ToeBase']]['position'].copy()
                elif 'leg_D_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_UpperLeg']]['position'].copy()
                    bone.effect_index = bone_name_dict[f'J_Bip_{node_name[-1]}_UpperLeg']['index']
                    bone.effect_factor = 1
                    bone.layer = 1
                elif 'knee_D_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_LowerLeg']]['position'].copy()
                    bone.effect_index = bone_name_dict[f'J_Bip_{node_name[-1]}_LowerLeg']['index']
                    bone.effect_factor = 1
                    bone.layer = 1
                elif 'ankle_D_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_Foot']]['position'].copy()
                    bone.effect_index = bone_name_dict[f'J_Bip_{node_name[-1]}_Foot']['index']
                    bone.effect_factor = 1
                    bone.layer = 1
                elif 'toe_EX_' in node_name:
                    position = node_dict[node_name_dict[f'J_Bip_{node_name[-1]}_ToeBase']]['position'].copy()
                    bone.layer = 1
            bone.position = position
            bone.index = len(model.bones)

            # 表示枠
            if bone_param["display"]:
                if bone_param["display"] not in model.display_slots:
                    model.display_slots[bone_param["display"]] = DisplaySlot(bone_param["display"], bone_param["display"], 0, 0)
                model.display_slots[bone_param["display"]].references.append((0, bone.index))

            model.bones[bone.name] = bone
            bone_name_dict[node_name] = {'index': bone.index, 'name': bone.name, 'node_name': node_name, 'node_index': node_index}
        
        if "髪" not in model.display_slots:
            model.display_slots["髪"] = DisplaySlot("髪", "Hair", 0, 0)
        if "その他" not in model.display_slots:
            model.display_slots["その他"] = DisplaySlot("その他", "Other", 0, 0)

        # 人体以外のボーン
        hair_blocks = []
        other_blocks = []
        for nidx, node_param in node_dict.items():
            if node_param['name'] not in bone_name_dict:
                bone = Bone(node_param['name'], node_param['name'], node_param['position'], -1, 0, 0x0002)
                parent_index = bone_name_dict[node_dict[node_param['parent']]['name']]['index'] if node_param['parent'] in node_dict and node_dict[node_param['parent']]['name'] in bone_name_dict else -1   # noqa
                bone.parent_index = parent_index
                bone.index = len(model.bones)
                
                if node_param['name'] not in DISABLE_BONES:
                    node_names = node_param['name'].split('_') if "Hair" in node_param['name'] else node_param['name'].split('_J_')
                    bone_block = None
                    bone_name = None

                    if "Hair" in node_param['name']:
                        if len(hair_blocks) == 0:
                            bone_block = {"bone_block_name": "髪", "bone_block_size": 1, "size": 1}
                        else:
                            bone_block = {"bone_block_name": "髪", "bone_block_size": hair_blocks[-1]["bone_block_size"], "size": hair_blocks[-1]["size"] + 1}
                        hair_blocks.append(bone_block)
                    else:
                        if len(other_blocks) == 0:
                            bone_block = {"bone_block_name": "装飾", "bone_block_size": 1, "size": 1}
                        else:
                            bone_block = {"bone_block_name": "装飾", "bone_block_size": other_blocks[-1]["bone_block_size"], "size": other_blocks[-1]["size"] + 1}
                        other_blocks.append(bone_block)
                    bone_name = f'{bone_block["bone_block_name"]}_{bone_block["bone_block_size"]:02d}-{bone_block["size"]:02d}'

                    if "Hair" not in node_param['name'] and len(node_names) > 1:
                        # 装飾の場合、末尾を入れる
                        bone_name += node_param['name'][len(node_names[0]):]

                    bone.name = bone_name

                model.bones[bone.name] = bone
                bone_name_dict[node_param['name']] = {'index': bone.index, 'name': bone.name, 'node_name': node_param['name'], 'node_index': node_name_dict[node_param['name']]}

                if node_param['name'] not in DISABLE_BONES:
                    if len(node_param['children']) == 0:
                        # 末端の場合次ボーンで段を変える(加算用にsizeは0)
                        if "Hair" in node_param['name']:
                            hair_blocks.append({"bone_block_name": "髪", "bone_block_size": hair_blocks[-1]["bone_block_size"] + 1, "size": 0})
                        else:
                            other_blocks.append({"bone_block_name": "装飾", "bone_block_size": other_blocks[-1]["bone_block_size"] + 1, "size": 0})

        # ローカル軸・IK設定
        for bone in model.bones.values():
            model.bone_indexes[bone.index] = bone.name

            # 人体ボーン
            if bone.english_name in BONE_PAIRS:
                # 表示先
                tail = BONE_PAIRS[bone.english_name]['tail']
                if tail:
                    if type(tail) is MVector3D:
                        bone.tail_position = tail.copy()
                    else:
                        bone.tail_index = bone_name_dict[tail]['index']
                if bone.name == '下半身':
                    # 腰は表示順が上なので、相対指定
                    bone.tail_position = model.bones['腰'].position - bone.position

                direction = bone.name[0]

                # 足IK
                leg_name = f'{direction}足'
                knee_name = f'{direction}ひざ'
                ankle_name = f'{direction}足首'
                toe_name = f'{direction}つま先'

                if bone.name in ['右足ＩＫ', '左足ＩＫ'] and leg_name in model.bones and knee_name in model.bones and ankle_name in model.bones:
                    leg_ik_link = []
                    leg_ik_link.append(IkLink(model.bones[knee_name].index, 1, MVector3D(math.radians(-180), 0, 0), MVector3D(math.radians(-0.5), 0, 0)))
                    leg_ik_link.append(IkLink(model.bones[leg_name].index, 0))
                    leg_ik = Ik(model.bones[ankle_name].index, 40, 1, leg_ik_link)
                    bone.ik = leg_ik

                if bone.name in ['右つま先ＩＫ', '左つま先ＩＫ'] and ankle_name in model.bones and toe_name in model.bones:
                    toe_ik_link = []
                    toe_ik_link.append(IkLink(model.bones[ankle_name].index, 0))
                    toe_ik = Ik(model.bones[toe_name].index, 40, 1, toe_ik_link)
                    bone.ik = toe_ik

                if bone.name in ['右目', '左目'] and '両目' in model.bones:
                    bone.flag |= 0x0100
                    bone.effect_index = model.bones['両目'].index
                    bone.effect_factor = 0.3
            else:
                # 人体以外
                # 表示先
                node_param = node_dict[node_name_dict[bone.english_name]]
                tail_index = bone_name_dict[node_dict[node_param['children'][0]]['name']]['index'] if node_param['children'] and node_param['children'][0] in node_dict and node_dict[node_param['children'][0]]['name'] in bone_name_dict else -1   # noqa
                if tail_index >= 0:
                    bone.tail_index = tail_index
                    bone.flag |= 0x0001 | 0x0008 | 0x0010

                if "Hair" in bone.english_name:
                    if bone.tail_index >= 0:
                        model.display_slots["髪"].references.append((0, bone.index))
                else:
                    model.display_slots["その他"].references.append((0, bone.index))

        logger.info("-- ボーンデータ解析終了")

        return model, bone_name_dict
    
    def calc_bone_position(self, model: PmxModel, node_dict: dict, node_param: dict):
        if node_param['parent'] == -1:
            return node_param['relative_position']

        return node_param['relative_position'] + self.calc_bone_position(model, node_dict, node_dict[node_param['parent']])

    def create_model(self):
        model = PmxModel()

        # テクスチャ用ディレクトリ
        tex_dir_path = os.path.join(str(Path(self.options.output_path).resolve().parents[0]), "tex")
        os.makedirs(tex_dir_path, exist_ok=True)
        # 展開用ディレクトリ作成
        glft_dir_path = os.path.join(str(Path(self.options.output_path).resolve().parents[0]), "glTF")
        os.makedirs(glft_dir_path, exist_ok=True)

        with open(self.options.pmx_model.path, "rb") as f:
            self.buffer = f.read()

            signature = self.unpack(12, "12s")
            logger.test("signature: %s (%s)", signature, self.offset)

            # JSON文字列読み込み
            json_buf_size = self.unpack(8, "L")
            json_text = self.read_text(json_buf_size)

            model.json_data = json.loads(json_text)
            
            # JSON出力
            jf = open(os.path.join(glft_dir_path, "gltf.json"), "w", encoding='utf-8')
            json.dump(model.json_data, jf, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
            logger.info("-- JSON出力終了")

            if "extensions" not in model.json_data or 'VRM' not in model.json_data['extensions'] or 'exporterVersion' not in model.json_data['extensions']['VRM']:
                logger.error("出力ソフト情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
                return None, None

            if "extensions" not in model.json_data or 'VRM' not in model.json_data['extensions'] or 'meta' not in model.json_data['extensions']['VRM']:
                logger.error("メタ情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
                return None, None

            if "VRoidStudio-0." in model.json_data['extensions']['VRM']['exporterVersion']:
                # VRoid Studioベータ版はNG
                logger.error("VRoid Studio ベータ版 で出力されたvrmデータではあるため、処理を中断します。\n正式版でコンバートしてから再度試してください。\n出力元: %s", \
                             model.json_data['extensions']['VRM']['exporterVersion'], decoration=MLogger.DECORATION_BOX)
                return None, None

            if "VRoid Studio-1." not in model.json_data['extensions']['VRM']['exporterVersion']:
                # VRoid Studio正式版じゃなくても警告だけに留める
                logger.warning("VRoid Studio 1.x で出力されたvrmデータではないため、結果がおかしくなる可能性があります。\n（結果がおかしくてもサポート対象外となります）\n出力元: %s", \
                               model.json_data['extensions']['VRM']['exporterVersion'], decoration=MLogger.DECORATION_BOX)

            if 'title' in model.json_data['extensions']['VRM']['meta']:
                model.name = model.json_data['extensions']['VRM']['meta']['title']
                model.english_name = model.json_data['extensions']['VRM']['meta']['title']
            if not model.name:
                # titleにモデル名が入ってなかった場合、ファイル名を代理入力
                file_name = os.path.basename(self.options.pmx_model.path).split('.')[0]
                model.name = file_name
                model.english_name = file_name

            model.comment += f"{logger.transtext('PMX出力')}: Vroid2Pmx\r\n"

            model.comment += f"\r\n{logger.transtext('アバター情報')} -------\r\n"

            if 'author' in model.json_data['extensions']['VRM']['meta']:
                model.comment += f"{logger.transtext('作者')}: {model.json_data['extensions']['VRM']['meta']['author']}\r\n"
            if 'contactInformation' in model.json_data['extensions']['VRM']['meta']:
                model.comment += f"{logger.transtext('連絡先')}: {model.json_data['extensions']['VRM']['meta']['contactInformation']}\r\n"
            if 'reference' in model.json_data['extensions']['VRM']['meta']:
                model.comment += f"{logger.transtext('参照')}: {model.json_data['extensions']['VRM']['meta']['reference']}\r\n"
            if 'version' in model.json_data['extensions']['VRM']['meta']:
                model.comment += f"{logger.transtext('バージョン')}: {model.json_data['extensions']['VRM']['meta']['version']}\r\n"

            model.comment += f"\r\n{logger.transtext('アバターの人格に関する許諾範囲')} -------\r\n"

            if 'allowedUserName' in model.json_data['extensions']['VRM']['meta']:
                model.comment += f"{logger.transtext('アバターに人格を与えることの許諾範囲')}: {model.json_data['extensions']['VRM']['meta']['allowedUserName']}\r\n"
            if 'violentUssageName' in model.json_data['extensions']['VRM']['meta']:
                model.comment += f"{logger.transtext('このアバターを用いて暴力表現を演じることの許可')}: {model.json_data['extensions']['VRM']['meta']['violentUssageName']}\r\n"
            if 'sexualUssageName' in model.json_data['extensions']['VRM']['meta']:
                model.comment += f"{logger.transtext('このアバターを用いて性的表現を演じることの許可')}: {model.json_data['extensions']['VRM']['meta']['sexualUssageName']}\r\n"
            if 'commercialUssageName' in model.json_data['extensions']['VRM']['meta']:
                model.comment += f"{logger.transtext('商用利用の許可')}: {model.json_data['extensions']['VRM']['meta']['commercialUssageName']}\r\n"
            if 'otherPermissionUrl' in model.json_data['extensions']['VRM']['meta']:
                model.comment += f"{logger.transtext('その他のライセンス条件')}: {model.json_data['extensions']['VRM']['meta']['otherPermissionUrl']}\r\n"

            model.comment += f"\r\n{logger.transtext('再配布・改変に関する許諾範囲')} -------\r\n"

            if 'licenseName' in model.json_data['extensions']['VRM']['meta']:
                model.comment += f"{logger.transtext('ライセンスタイプ')}: {model.json_data['extensions']['VRM']['meta']['licenseName']}\r\n"
            if 'otherPermissionUrl' in model.json_data['extensions']['VRM']['meta']:
                model.comment += f"{logger.transtext('その他のライセンス条件')}: {model.json_data['extensions']['VRM']['meta']['otherPermissionUrl']}\r\n"

            # binデータ
            bin_buf_size = self.unpack(8, "L")
            logger.debug(f'bin_buf_size: {bin_buf_size}')

            with open(os.path.join(glft_dir_path, "data.bin"), "wb") as bf:
                bf.write(self.buffer[self.offset:(self.offset + bin_buf_size)])

            # 空値をスフィア用に登録
            model.textures.append("")

            if "images" not in model.json_data:
                logger.error("変換可能な画像情報がないため、処理を中断します。", decoration=MLogger.DECORATION_BOX)
                return None, None

            # jsonデータの中に画像データの指定がある場合
            image_offset = 0
            for image in model.json_data['images']:
                if int(image["bufferView"]) < len(model.json_data['bufferViews']):
                    image_buffer = model.json_data['bufferViews'][int(image["bufferView"])]
                    # 画像の開始位置はオフセット分ずらす
                    image_start = self.offset + image_buffer["byteOffset"]
                    # 拡張子
                    ext = MIME_TYPE[image["mimeType"]]
                    # 画像名
                    image_name = f"{image['name']}.{ext}"
                    with open(os.path.join(glft_dir_path, image_name), "wb") as ibf:
                        ibf.write(self.buffer[image_start:(image_start + image_buffer["byteLength"])])
                    # オフセット加算
                    image_offset += image_buffer["byteLength"]
                    # PMXに追記
                    model.textures.append(os.path.join("tex", image_name))
                    # テクスチャコピー
                    shutil.copy(os.path.join(glft_dir_path, image_name), os.path.join(tex_dir_path, image_name))
            
            logger.info("-- テクスチャデータ解析終了")

        return model, tex_dir_path

    # アクセサ経由で値を取得する
    # https://github.com/ft-lab/Documents_glTF/blob/master/structure.md
    def read_from_accessor(self, model: PmxModel, accessor_idx: int):
        bresult = None
        aidx = 0
        if accessor_idx < len(model.json_data['accessors']):
            accessor = model.json_data['accessors'][accessor_idx]
            acc_type = accessor['type']
            if accessor['bufferView'] < len(model.json_data['bufferViews']):
                buffer = model.json_data['bufferViews'][accessor['bufferView']]
                logger.debug('accessor: %s, %s', accessor_idx, buffer)
                if 'count' in accessor:
                    bresult = []
                    if acc_type == "VEC3":
                        buf_type, buf_num = self.define_buf_type(accessor['componentType'])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor['count']):
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
                        buf_type, buf_num = self.define_buf_type(accessor['componentType'])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor['count']):
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
                        buf_type, buf_num = self.define_buf_type(accessor['componentType'])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor['count']):
                            buf_start = self.offset + buffer["byteOffset"] + ((buf_num * 4) * n)

                            # Vec3 / float
                            xresult = struct.unpack_from(buf_type, self.buffer, buf_start)
                            yresult = struct.unpack_from(buf_type, self.buffer, buf_start + buf_num)
                            zresult = struct.unpack_from(buf_type, self.buffer, buf_start + (buf_num * 2))
                            wresult = struct.unpack_from(buf_type, self.buffer, buf_start + (buf_num * 3))

                            if buf_type == "f":
                                bresult.append(MVector4D(float(xresult[0]), float(yresult[0]), float(zresult[0]), float(wresult[0])))
                            else:
                                bresult.append(MVector4D(int(xresult[0]), int(yresult[0]), int(zresult[0]), int(wresult[0])))
                            
                            aidx += 1

                            if aidx % 5000 == 0:
                                logger.info("-- -- Accessor[%s/%s/%s][%s]", accessor_idx, acc_type, buf_type, aidx)
                            else:
                                logger.test("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                    elif acc_type == "SCALAR":
                        buf_type, buf_num = self.define_buf_type(accessor['componentType'])
                        if accessor_idx % 10 == 0:
                            logger.info("-- -- Accessor[%s/%s/%s]", accessor_idx, acc_type, buf_type)

                        for n in range(accessor['count']):
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


DISABLE_BONES = [
    'Face',
    'Body',
    'Hairs',
    'Hair001',
    'secondary',
]

BONE_PAIRS = {
    'Root': {'name': '全ての親', 'parent': None, 'tail': 'Center', 'display': None, 'flag': 0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0010, \
             'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'Center': {'name': 'センター', 'parent': 'Root', 'tail': None, 'display': 'センター', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010, \
               'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'Groove': {'name': 'グルーブ', 'parent': 'Center', 'tail': None, 'display': 'センター', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010, \
               'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_C_Hips': {'name': '腰', 'parent': 'Groove', 'tail': None, 'display': '体幹', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010, \
                     'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_C_Spine': {'name': '下半身', 'parent': 'J_Bip_C_Hips', 'tail': None, 'display': '体幹', 'flag': 0x0002 | 0x0008 | 0x0010, \
                      'rigidbodyGroup': 0, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_C_Spine2': {'name': '上半身', 'parent': 'J_Bip_C_Hips', 'tail': 'J_Bip_C_Chest', 'display': '体幹', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                       'rigidbodyGroup': 0, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_C_Chest': {'name': '上半身2', 'parent': 'J_Bip_C_Spine2', 'tail': 'J_Bip_C_UpperChest', 'display': '体幹', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                      'rigidbodyGroup': 0, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_C_UpperChest': {'name': '上半身3', 'parent': 'J_Bip_C_Chest', 'tail': 'J_Bip_C_Neck', 'display': '体幹', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                           'rigidbodyGroup': 0, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_C_Neck': {'name': '首', 'parent': 'J_Bip_C_UpperChest', 'tail': 'J_Bip_C_Head', 'display': '体幹', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                     'rigidbodyGroup': 0, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_C_Head': {'name': '頭', 'parent': 'J_Bip_C_Neck', 'tail': None, 'display': '体幹', 'flag': 0x0002 | 0x0008 | 0x0010, \
                     'rigidbodyGroup': 0, 'rigidbodyShape': 0, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Adj_FaceEye': {'name': '両目', 'parent': 'J_Bip_C_Head', 'tail': None, 'display': '顔', 'flag': 0x0002 | 0x0008 | 0x0010, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Adj_L_FaceEye': {'name': '左目', 'parent': 'J_Bip_C_Head', 'tail': None, 'display': '顔', 'flag': 0x0002 | 0x0008 | 0x0010, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Adj_R_FaceEye': {'name': '右目', 'parent': 'J_Bip_C_Head', 'tail': None, 'display': '顔', 'flag': 0x0002 | 0x0008 | 0x0010, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Sec_L_Bust1': {'name': '左胸', 'parent': 'J_Bip_C_UpperChest', 'tail': 'J_Sec_L_Bust2', 'display': '胸', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                      'rigidbodyGroup': 0, 'rigidbodyShape': 0, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Sec_L_Bust2': {'name': '左胸先', 'parent': 'J_Sec_L_Bust1', 'tail': None, 'display': None, 'flag': 0x0002, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Sec_R_Bust1': {'name': '右胸', 'parent': 'J_Bip_C_UpperChest', 'tail': 'J_Sec_R_Bust2', 'display': '胸', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                      'rigidbodyGroup': 0, 'rigidbodyShape': 0, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Sec_R_Bust2': {'name': '右胸先', 'parent': 'J_Sec_R_Bust1', 'tail': None, 'display': None, 'flag': 0x0002, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'shoulderP_L': {'name': '左肩P', 'parent': 'J_Bip_C_UpperChest', 'tail': None, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010, \
                    'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Shoulder': {'name': '左肩', 'parent': 'shoulderP_L', 'tail': 'J_Bip_L_UpperArm', 'display': '左手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                         'rigidbodyGroup': 2, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'shoulderC_L': {'name': '左肩C', 'parent': 'J_Bip_L_Shoulder', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                    'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_UpperArm': {'name': '左腕', 'parent': 'shoulderC_L', 'tail': 'J_Bip_L_LowerArm', 'display': '左手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                         'rigidbodyGroup': 2, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'arm_twist_L': {'name': '左腕捩', 'parent': 'J_Bip_L_UpperArm', 'tail': None, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800 | 0x0800, \
                    'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'arm_twist_1L': {'name': '左腕捩1', 'parent': 'J_Bip_L_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                     'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'arm_twist_2L': {'name': '左腕捩2', 'parent': 'J_Bip_L_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                     'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'arm_twist_3L': {'name': '左腕捩3', 'parent': 'J_Bip_L_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                     'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_LowerArm': {'name': '左ひじ', 'parent': 'arm_twist_L', 'tail': 'J_Bip_L_Hand', 'display': '左手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                         'rigidbodyGroup': 2, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'wrist_twist_L': {'name': '左手捩', 'parent': 'J_Bip_L_LowerArm', 'tail': None, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'wrist_twist_1L': {'name': '左手捩1', 'parent': 'J_Bip_L_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'wrist_twist_2L': {'name': '左手捩2', 'parent': 'J_Bip_L_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'wrist_twist_3L': {'name': '左手捩3', 'parent': 'J_Bip_L_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Hand': {'name': '左手首', 'parent': 'wrist_twist_L', 'tail': None, 'display': '左手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                     'rigidbodyGroup': 2, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_L_Thumb1': {'name': '左親指０', 'parent': 'J_Bip_L_Hand', 'tail': 'J_Bip_L_Thumb2', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Thumb2': {'name': '左親指１', 'parent': 'J_Bip_L_Thumb1', 'tail': 'J_Bip_L_Thumb3', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Thumb3': {'name': '左親指２', 'parent': 'J_Bip_L_Thumb2', 'tail': 'J_Bip_L_Thumb3_end', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Thumb3_end': {'name': '左親指先', 'parent': 'J_Bip_L_Thumb3', 'tail': None, 'display': None, 'flag': 0x0002, \
                           'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Index1': {'name': '左人指１', 'parent': 'J_Bip_L_Hand', 'tail': 'J_Bip_L_Index2', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Index2': {'name': '左人指２', 'parent': 'J_Bip_L_Index1', 'tail': 'J_Bip_L_Index3', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Index3': {'name': '左人指３', 'parent': 'J_Bip_L_Index2', 'tail': 'J_Bip_L_Index3_end', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Index3_end': {'name': '左人指先', 'parent': 'J_Bip_L_Index3', 'tail': None, 'display': None, 'flag': 0x0002, \
                           'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Middle1': {'name': '左中指１', 'parent': 'J_Bip_L_Hand', 'tail': 'J_Bip_L_Middle2', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Middle2': {'name': '左中指２', 'parent': 'J_Bip_L_Middle1', 'tail': 'J_Bip_L_Middle3', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Middle3': {'name': '左中指３', 'parent': 'J_Bip_L_Middle2', 'tail': 'J_Bip_L_Middle3_end', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Middle3_end': {'name': '左中指先', 'parent': 'J_Bip_L_Middle3', 'tail': None, 'display': None, 'flag': 0x0002, \
                            'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Ring1': {'name': '左薬指１', 'parent': 'J_Bip_L_Hand', 'tail': 'J_Bip_L_Ring2', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Ring2': {'name': '左薬指２', 'parent': 'J_Bip_L_Ring1', 'tail': 'J_Bip_L_Ring3', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Ring3': {'name': '左薬指３', 'parent': 'J_Bip_L_Ring2', 'tail': 'J_Bip_L_Ring3_end', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Ring3_end': {'name': '左薬指先', 'parent': 'J_Bip_L_Ring3', 'tail': None, 'display': None, 'flag': 0x0002, \
                          'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Little1': {'name': '左小指１', 'parent': 'J_Bip_L_Hand', 'tail': 'J_Bip_L_Little2', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Little2': {'name': '左小指２', 'parent': 'J_Bip_L_Little1', 'tail': 'J_Bip_L_Little3', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Little3': {'name': '左小指３', 'parent': 'J_Bip_L_Little2', 'tail': 'J_Bip_L_Little3_end', 'display': '左指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_Little3_end': {'name': '左小指先', 'parent': 'J_Bip_L_Little3', 'tail': None, 'display': None, 'flag': 0x0002, \
                            'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'shoulderP_R': {'name': '右肩P', 'parent': 'J_Bip_C_UpperChest', 'tail': None, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010, \
                    'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Shoulder': {'name': '右肩', 'parent': 'shoulderP_R', 'tail': 'J_Bip_R_UpperArm', 'display': '右手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                         'rigidbodyGroup': 2, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'shoulderC_R': {'name': '右肩C', 'parent': 'J_Bip_R_Shoulder', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                    'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_UpperArm': {'name': '右腕', 'parent': 'shoulderC_R', 'tail': 'J_Bip_R_LowerArm', 'display': '右手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                         'rigidbodyGroup': 2, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'arm_twist_R': {'name': '右腕捩', 'parent': 'J_Bip_R_UpperArm', 'tail': None, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800, \
                    'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'arm_twist_1R': {'name': '右腕捩1', 'parent': 'J_Bip_R_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                     'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'arm_twist_2R': {'name': '右腕捩2', 'parent': 'J_Bip_R_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                     'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'arm_twist_3R': {'name': '右腕捩3', 'parent': 'J_Bip_R_UpperArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                     'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_LowerArm': {'name': '右ひじ', 'parent': 'arm_twist_R', 'tail': 'J_Bip_R_Hand', 'display': '右手', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                         'rigidbodyGroup': 2, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'wrist_twist_R': {'name': '右手捩', 'parent': 'J_Bip_R_LowerArm', 'tail': None, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0400 | 0x0800, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'wrist_twist_1R': {'name': '右手捩1', 'parent': 'J_Bip_R_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'wrist_twist_2R': {'name': '右手捩2', 'parent': 'J_Bip_R_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'wrist_twist_3R': {'name': '右手捩3', 'parent': 'J_Bip_R_LowerArm', 'tail': None, 'display': None, 'flag': 0x0002 | 0x0100, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Hand': {'name': '右手首', 'parent': 'wrist_twist_R', 'tail': None, 'display': '右手', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                     'rigidbodyGroup': 2, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_R_Thumb1': {'name': '右親指０', 'parent': 'J_Bip_R_Hand', 'tail': 'J_Bip_R_Thumb2', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Thumb2': {'name': '右親指１', 'parent': 'J_Bip_R_Thumb1', 'tail': 'J_Bip_R_Thumb3', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Thumb3': {'name': '右親指２', 'parent': 'J_Bip_R_Thumb2', 'tail': 'J_Bip_R_Thumb3_end', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Thumb3_end': {'name': '右親指先', 'parent': 'J_Bip_R_Thumb3', 'tail': None, 'display': None, 'flag': 0x0002, \
                           'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Index1': {'name': '右人指１', 'parent': 'J_Bip_R_Hand', 'tail': 'J_Bip_R_Index2', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Index2': {'name': '右人指２', 'parent': 'J_Bip_R_Index1', 'tail': 'J_Bip_R_Index3', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Index3': {'name': '右人指３', 'parent': 'J_Bip_R_Index2', 'tail': 'J_Bip_R_Index3_end', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                       'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Index3_end': {'name': '右人指先', 'parent': 'J_Bip_R_Index3', 'tail': None, 'display': None, 'flag': 0x0002, \
                           'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Middle1': {'name': '右中指１', 'parent': 'J_Bip_R_Hand', 'tail': 'J_Bip_R_Middle2', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Middle2': {'name': '右中指２', 'parent': 'J_Bip_R_Middle1', 'tail': 'J_Bip_R_Middle3', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Middle3': {'name': '右中指３', 'parent': 'J_Bip_R_Middle2', 'tail': 'J_Bip_R_Middle3_end', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Middle3_end': {'name': '右中指先', 'parent': 'J_Bip_R_Middle3', 'tail': None, 'display': None, 'flag': 0x0002, \
                            'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Ring1': {'name': '右薬指１', 'parent': 'J_Bip_R_Hand', 'tail': 'J_Bip_R_Ring2', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Ring2': {'name': '右薬指２', 'parent': 'J_Bip_R_Ring1', 'tail': 'J_Bip_R_Ring3', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Ring3': {'name': '右薬指３', 'parent': 'J_Bip_R_Ring2', 'tail': 'J_Bip_R_Ring3_end', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Ring3_end': {'name': '右薬指先', 'parent': 'J_Bip_R_Ring3', 'tail': None, 'display': None, 'flag': 0x0002, \
                          'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Little1': {'name': '右小指１', 'parent': 'J_Bip_R_Hand', 'tail': 'J_Bip_R_Little2', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Little2': {'name': '右小指２', 'parent': 'J_Bip_R_Little1', 'tail': 'J_Bip_R_Little3', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Little3': {'name': '右小指３', 'parent': 'J_Bip_R_Little2', 'tail': 'J_Bip_R_Little3_end', 'display': '右指', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010 | 0x0800, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_Little3_end': {'name': '右小指先', 'parent': 'J_Bip_R_Little3', 'tail': None, 'display': None, 'flag': 0x0002, \
                            'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'waistCancel_L': {'name': '腰キャンセル左', 'parent': 'J_Bip_C_Spine', 'tail': None, 'display': None, 'flag': 0x0002, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_L_UpperLeg': {'name': '左足', 'parent': 'waistCancel_L', 'tail': 'J_Bip_L_LowerLeg', 'display': '左足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                         'rigidbodyGroup': 1, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_L_LowerLeg': {'name': '左ひざ', 'parent': 'J_Bip_L_UpperLeg', 'tail': 'J_Bip_L_Foot', 'display': '左足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                         'rigidbodyGroup': 1, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_L_Foot': {'name': '左足首', 'parent': 'J_Bip_L_LowerLeg', 'tail': 'J_Bip_L_ToeBase', 'display': '左足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                     'rigidbodyGroup': 1, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_L_ToeBase': {'name': '左つま先', 'parent': 'J_Bip_L_Foot', 'tail': None, 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'leg_IK_Parent_L': {'name': '左足IK親', 'parent': 'Root', 'tail': 'leg_IK_L', 'display': '左足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'leg_IK_L': {'name': '左足ＩＫ', 'parent': 'leg_IK_Parent_L', 'tail': MVector3D(0, 0, 1), 'display': '左足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020, \
                 'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'toe_IK_L': {'name': '左つま先ＩＫ', 'parent': 'leg_IK_L', 'tail': MVector3D(0, -1, 0), 'display': '左足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020, \
                 'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'waistCancel_R': {'name': '腰キャンセル右', 'parent': 'J_Bip_C_Spine', 'tail': None, 'display': None, 'flag': 0x0002, \
                      'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'J_Bip_R_UpperLeg': {'name': '右足', 'parent': 'waistCancel_R', 'tail': 'J_Bip_R_LowerLeg', 'display': '右足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                         'rigidbodyGroup': 1, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_R_LowerLeg': {'name': '右ひざ', 'parent': 'J_Bip_R_UpperLeg', 'tail': 'J_Bip_R_Foot', 'display': '右足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                         'rigidbodyGroup': 1, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_R_Foot': {'name': '右足首', 'parent': 'J_Bip_R_LowerLeg', 'tail': 'J_Bip_R_ToeBase', 'display': '右足', 'flag': 0x0001 | 0x0002 | 0x0008 | 0x0010, \
                     'rigidbodyGroup': 1, 'rigidbodyShape': 2, 'rigidbodyMode': 0, 'rigidbodyNoColl': [0, 1, 2]},
    'J_Bip_R_ToeBase': {'name': '右つま先', 'parent': 'J_Bip_R_Foot', 'tail': None, 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'leg_IK_Parent_R': {'name': '右足IK親', 'parent': 'Root', 'tail': 'leg_IK_R', 'display': '右足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010, \
                        'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'leg_IK_R': {'name': '右足ＩＫ', 'parent': 'leg_IK_Parent_R', 'tail': MVector3D(0, 0, 1), 'display': '右足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020, \
                 'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'toe_IK_R': {'name': '右つま先ＩＫ', 'parent': 'leg_IK_R', 'tail': MVector3D(0, -1, 0), 'display': '右足', 'flag': 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020, \
                 'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'leg_D_L': {'name': '左足D', 'parent': 'waistCancel_L', 'tail': None, 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100, \
                'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'knee_D_L': {'name': '左ひざD', 'parent': 'leg_D_L', 'tail': None, 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100, \
                 'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'ankle_D_L': {'name': '左足首D', 'parent': 'knee_D_L', 'tail': None, 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100, \
                  'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'toe_EX_L': {'name': '左足先EX', 'parent': 'ankle_D_L', 'tail': MVector3D(0, 0, -1), 'display': '左足', 'flag': 0x0002 | 0x0008 | 0x0010, \
                 'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'leg_D_R': {'name': '右足D', 'parent': 'waistCancel_R', 'tail': None, 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100, \
                'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'knee_D_R': {'name': '右ひざD', 'parent': 'leg_D_R', 'tail': None, 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100, \
                 'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'ankle_D_R': {'name': '右足首D', 'parent': 'knee_D_R', 'tail': None, 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010 | 0x0100, \
                  'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
    'toe_EX_R': {'name': '右足先EX', 'parent': 'ankle_D_R', 'tail': MVector3D(0, 0, -1), 'display': '右足', 'flag': 0x0002 | 0x0008 | 0x0010, \
                 'rigidbodyGroup': -1, 'rigidbodyShape': -1, 'rigidbodyMode': 0, 'rigidbodyNoColl': None},
}

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
    "Fcl_BRW_Fun": {"name": "にこり", "panel": MORPH_EYEBROW},
    "Fcl_BRW_Fun_R": {"name": "にこり右", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Fun"},
    "Fcl_BRW_Fun_L": {"name": "にこり左", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Fun"},
    "Fcl_BRW_Joy": {"name": "にこり2", "panel": MORPH_EYEBROW},
    "Fcl_BRW_Joy_R": {"name": "にこり2右", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Joy"},
    "Fcl_BRW_Joy_L": {"name": "にこり2左", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Joy"},
    "Fcl_BRW_Sorrow": {"name": "困り", "panel": MORPH_EYEBROW},
    "Fcl_BRW_Sorrow_R": {"name": "困り右", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Sorrow"},
    "Fcl_BRW_Sorrow_L": {"name": "困り左", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Sorrow"},
    "Fcl_BRW_Angry": {"name": "怒り", "panel": MORPH_EYEBROW},
    "Fcl_BRW_Angry_R": {"name": "怒り右", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Angry"},
    "Fcl_BRW_Angry_L": {"name": "怒り左", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Angry"},
    "Fcl_BRW_Surprised": {"name": "驚き", "panel": MORPH_EYEBROW},
    "Fcl_BRW_Surprised_R": {"name": "驚き右", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Surprised"},
    "Fcl_BRW_Surprised_L": {"name": "驚き左", "panel": MORPH_EYEBROW, "split": "Fcl_BRW_Surprised"},
    "browInnerUp": {"name": "ひそめる", "panel": MORPH_EYEBROW},
    "browInnerUp_R": {"name": "ひそめる右", "panel": MORPH_EYEBROW, "split": "browInnerUp"},
    "browInnerUp_L": {"name": "ひそめる左", "panel": MORPH_EYEBROW, "split": "browInnerUp"},
    "browDown": {"name": "真面目", "panel": MORPH_EYEBROW, "binds": ["browDownRight", "browDownLeft"]},
    "browDownRight": {"name": "真面目右", "panel": MORPH_EYEBROW},
    "browDownLeft": {"name": "真面目左", "panel": MORPH_EYEBROW},
    "browOuter": {"name": "はんっ", "panel": MORPH_EYEBROW, "binds": ["browOuterUpRight", "browOuterUpLeft"]},
    "browOuterUpRight": {"name": "はんっ右", "panel": MORPH_EYEBROW},
    "browOuterUpLeft": {"name": "はんっ左", "panel": MORPH_EYEBROW},

    "Fcl_EYE_Natural": {"name": "ナチュラル", "panel": MORPH_EYE},
    "Fcl_EYE_Close": {"name": "まばたき", "panel": MORPH_EYE},
    "Fcl_EYE_Close_R": {"name": "まばたき右", "panel": MORPH_EYE},
    "Fcl_EYE_Close_L": {"name": "まばたき左", "panel": MORPH_EYE},
    "Fcl_EYE_Joy": {"name": "笑い", "panel": MORPH_EYE},
    "Fcl_EYE_Joy_L": {"name": "ウィンク", "panel": MORPH_EYE},
    "Fcl_EYE_Joy_R": {"name": "ウィンク右", "panel": MORPH_EYE},
    "Fcl_EYE_Fun": {"name": "喜び", "panel": MORPH_EYE},
    "Fcl_EYE_Fun_R": {"name": "喜び右", "panel": MORPH_EYE, "split": "Fcl_EYE_Fun"},
    "Fcl_EYE_Fun_L": {"name": "喜び左", "panel": MORPH_EYE, "split": "Fcl_EYE_Fun"},
    "eyeSquint": {"name": "にんまり", "panel": MORPH_EYE, "binds": ["eyeSquintRight", "eyeSquintLeft"]},
    "eyeSquintRight": {"name": "にんまり右", "panel": MORPH_EYE},
    "eyeSquintLeft": {"name": "にんまり左", "panel": MORPH_EYE},
    "Fcl_EYE_Angry": {"name": "キリッ", "panel": MORPH_EYE},
    "Fcl_EYE_Angry_R": {"name": "キリッ右", "panel": MORPH_EYE, "split": "Fcl_EYE_Angry"},
    "Fcl_EYE_Angry_L": {"name": "キリッ左", "panel": MORPH_EYE, "split": "Fcl_EYE_Angry"},
    "noseSneer": {"name": "キリッ2", "panel": MORPH_EYE, "binds": ["noseSneerRight", "noseSneerLeft"]},
    "noseSneerRight": {"name": "キリッ2右", "panel": MORPH_EYE},
    "noseSneerLeft": {"name": "キリッ2左", "panel": MORPH_EYE},
    "Fcl_EYE_Sorrow": {"name": "じと目", "panel": MORPH_EYE},
    "Fcl_EYE_Sorrow_R": {"name": "じと目右", "panel": MORPH_EYE, "split": "Fcl_EYE_Sorrow"},
    "Fcl_EYE_Sorrow_L": {"name": "じと目左", "panel": MORPH_EYE, "split": "Fcl_EYE_Sorrow"},
    "Fcl_EYE_Spread": {"name": "見開き", "panel": MORPH_EYE},
    "Fcl_EYE_Spread_R": {"name": "見開き右", "panel": MORPH_EYE, "split": "Fcl_EYE_Spread"},
    "Fcl_EYE_Spread_L": {"name": "見開き左", "panel": MORPH_EYE, "split": "Fcl_EYE_Spread"},
    "Fcl_EYE_Surprised": {"name": "びっくり", "panel": MORPH_EYE},
    "Fcl_EYE_Surprised_R": {"name": "びっくり右", "panel": MORPH_EYE, "split": "Fcl_EYE_Surprised"},
    "Fcl_EYE_Surprised_L": {"name": "びっくり左", "panel": MORPH_EYE, "split": "Fcl_EYE_Surprised"},
    "eyeWide": {"name": "びっくり2", "panel": MORPH_EYE, "binds": ["eyeSquintRight", "eyeSquintLeft"]},
    "eyeWideRight": {"name": "びっくり2右", "panel": MORPH_EYE},
    "eyeWideLeft": {"name": "びっくり2左", "panel": MORPH_EYE},

    "eyeLookUp": {"name": "目上", "panel": MORPH_EYE, "binds": ["eyeLookUpRight", "eyeLookUpLeft"]},
    "eyeLookUpRight": {"name": "目上右", "panel": MORPH_EYE},
    "eyeLookUpLeft": {"name": "目上左", "panel": MORPH_EYE},
    "eyeLookDown": {"name": "目下", "panel": MORPH_EYE, "binds": ["eyeLookDownRight", "eyeLookDownLeft"]},
    "eyeLookDownRight": {"name": "目下右", "panel": MORPH_EYE},
    "eyeLookDownLeft": {"name": "目下左", "panel": MORPH_EYE},
    "eyeLookIn": {"name": "目頭広", "panel": MORPH_EYE, "binds": ["eyeLookInRight", "eyeLookInLeft"]},
    "eyeLookInRight": {"name": "目頭広右", "panel": MORPH_EYE},
    "eyeLookInLeft": {"name": "目頭広左", "panel": MORPH_EYE},
    "eyeLookOut": {"name": "目尻広", "panel": MORPH_EYE, "binds": ["eyeLookOutRight", "eyeLookOutLeft"]},
    "eyeLookOutLeft": {"name": "目尻広右", "panel": MORPH_EYE},
    "eyeLookOutRight": {"name": "目尻広左", "panel": MORPH_EYE},
    # "eyeBlinkLeft": {"name": "", "panel": MORPH_EYE},
    # "eyeBlinkRight": {"name": "", "panel": MORPH_EYE},
    "_eyeIrisMoveBack": {"name": "瞳小", "panel": MORPH_EYE, "binds": ["_eyeIrisMoveBack_R", "_eyeIrisMoveBack_L"]},
    "_eyeIrisMoveBack_R": {"name": "瞳小右", "panel": MORPH_EYE},
    "_eyeIrisMoveBack_L": {"name": "瞳小左", "panel": MORPH_EYE},
    "_eyeSquint+LowerUp": {"name": "下瞼上げ", "panel": MORPH_EYE, "binds": ["_eyeSquint+LowerUp_R", "_eyeSquint+LowerUp_L"]},
    "_eyeSquint+LowerUp_R": {"name": "下瞼上げ右", "panel": MORPH_EYE},
    "_eyeSquint+LowerUp_L": {"name": "下瞼上げ左", "panel": MORPH_EYE},

    "Fcl_EYE_Iris_Hide": {"name": "白目", "panel": MORPH_EYE},
    "Fcl_EYE_Iris_Hide_R": {"name": "白目右", "panel": MORPH_EYE, "split": "Fcl_EYE_Iris_Hide"},
    "Fcl_EYE_Iris_Hide_L": {"name": "白目左", "panel": MORPH_EYE, "split": "Fcl_EYE_Iris_Hide"},
    "Fcl_EYE_Highlight_Hide": {"name": "ハイライトなし", "panel": MORPH_EYE},
    "Fcl_EYE_Highlight_Hide_R": {"name": "ハイライトなし右", "panel": MORPH_EYE, "split": "Fcl_EYE_Highlight_Hide"},
    "Fcl_EYE_Highlight_Hide_L": {"name": "ハイライトなし左", "panel": MORPH_EYE, "split": "Fcl_EYE_Highlight_Hide"},

    "Fcl_MTH_A": {"name": "あ", "panel": MORPH_LIP, "ratio": 0.7},
    "Fcl_MTH_I": {"name": "い", "panel": MORPH_LIP, "ratio": 0.7},
    "Fcl_MTH_U": {"name": "う", "panel": MORPH_LIP, "ratio": 0.7},
    "Fcl_MTH_E": {"name": "え", "panel": MORPH_LIP, "ratio": 0.7},
    "Fcl_MTH_O": {"name": "お", "panel": MORPH_LIP, "ratio": 0.7},
    "Fcl_MTH_Neutral": {"name": "口閉じ", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_MTH_Up": {"name": "口上", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_MTH_Down": {"name": "口下", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_MTH_Angry": {"name": "む", "panel": MORPH_LIP, "ratio": 0.7},
    "Fcl_MTH_Angry_R": {"name": "む右", "panel": MORPH_LIP, "ratio": 0.7, "split": "Fcl_MTH_Angry"},
    "Fcl_MTH_Angry_L": {"name": "む左", "panel": MORPH_LIP, "ratio": 0.7, "split": "Fcl_MTH_Angry"},
    "Fcl_MTH_Small": {"name": "すぼめる", "panel": MORPH_LIP, "ratio": 0.7},
    "Fcl_MTH_Large": {"name": "いー", "panel": MORPH_LIP, "ratio": 0.7},
    "Fcl_MTH_Fun": {"name": "にやり", "panel": MORPH_LIP, "ratio": 0.7},
    "Fcl_MTH_Fun_R": {"name": "にやり右", "panel": MORPH_LIP, "ratio": 0.7, "split": "Fcl_MTH_Fun"},
    "Fcl_MTH_Fun_L": {"name": "にやり左", "panel": MORPH_LIP, "ratio": 0.7, "split": "Fcl_MTH_Fun"},
    "Fcl_MTH_Joy": {"name": "ワ", "panel": MORPH_LIP, "ratio": 0.7},
    "Fcl_MTH_Sorrow": {"name": "△", "panel": MORPH_LIP, "ratio": 0.7},
    "Fcl_MTH_Surprised": {"name": "わー", "panel": MORPH_LIP, "ratio": 0.7},
    
    "jawOpen": {"name": "あああ", "panel": MORPH_LIP, "ratio": 1},
    "jawForward": {"name": "顎前", "panel": MORPH_LIP, "ratio": 1},
    "jawLeft": {"name": "顎左", "panel": MORPH_LIP, "ratio": 1},
    "jawRight": {"name": "顎右", "panel": MORPH_LIP, "ratio": 1},
    "mouthFunnel": {"name": "んむー", "panel": MORPH_LIP, "ratio": 1},
    "mouthPucker": {"name": "うー", "panel": MORPH_LIP, "ratio": 1},
    "mouthLeft": {"name": "口左", "panel": MORPH_LIP, "ratio": 1},
    "mouthRight": {"name": "口右", "panel": MORPH_LIP, "ratio": 1},
    "mouthRoll": {"name": "んむー", "panel": MORPH_LIP, "ratio": 1, "binds": ["mouthRollUpper", "mouthRollLower"]},
    "mouthRollUpper": {"name": "上唇んむー", "panel": MORPH_LIP, "ratio": 1},
    "mouthRollLower": {"name": "下唇んむー", "panel": MORPH_LIP, "ratio": 1},
    "mouthShrug": {"name": "むむ", "panel": MORPH_LIP, "ratio": 1, "binds": ["mouthShrugUpper", "mouthShrugLower"]},
    "mouthShrugUpper": {"name": "上唇むむ", "panel": MORPH_LIP, "ratio": 1},
    "mouthShrugLower": {"name": "下唇むむ", "panel": MORPH_LIP, "ratio": 1},
    # "mouthClose": {"name": "", "panel": MORPH_LIP, "ratio": 1},
    "mouthDimple": {"name": "口幅広", "panel": MORPH_LIP, "ratio": 1, "binds": ["mouthDimpleRight", "mouthDimpleLeft"]},
    "mouthDimpleRight": {"name": "口幅広右", "panel": MORPH_LIP, "ratio": 1},
    "mouthDimpleLeft": {"name": "口幅広左", "panel": MORPH_LIP, "ratio": 1},
    "mouthPress": {"name": "薄笑い", "panel": MORPH_LIP, "ratio": 1, "binds": ["mouthPressRight", "mouthPressLeft"]},
    "mouthPressRight": {"name": "薄笑い右", "panel": MORPH_LIP, "ratio": 1},
    "mouthPressLeft": {"name": "薄笑い左", "panel": MORPH_LIP, "ratio": 1},
    "mouthSmile": {"name": "にやり2", "panel": MORPH_LIP, "ratio": 1, "binds": ["mouthSmileRight", "mouthSmileLeft"]},
    "mouthSmileRight": {"name": "にやり2右", "panel": MORPH_LIP, "ratio": 1},
    "mouthSmileLeft": {"name": "にやり2左", "panel": MORPH_LIP, "ratio": 1},
    "mouthUpperUp": {"name": "にひ", "panel": MORPH_LIP, "ratio": 1, "binds": ["mouthUpperUpRight", "mouthDimpleLeft"]},
    "mouthUpperUpRight": {"name": "にひ右", "panel": MORPH_LIP, "ratio": 1},
    "mouthUpperUpLeft": {"name": "にひ左", "panel": MORPH_LIP, "ratio": 1},
    "cheekSquint": {"name": "にひひ", "panel": MORPH_LIP, "ratio": 1, "binds": ["cheekSquintRight", "cheekSquintLeft"]},
    "cheekSquintRight": {"name": "にひひ右", "panel": MORPH_LIP, "ratio": 1},
    "cheekSquintLeft": {"name": "にひひ左", "panel": MORPH_LIP, "ratio": 1},
    "mouthFrown": {"name": "ちっ", "panel": MORPH_LIP, "ratio": 1, "binds": ["mouthFrownRight", "mouthFrownLeft"]},
    "mouthFrownRight": {"name": "ちっ右", "panel": MORPH_LIP, "ratio": 1},
    "mouthFrownLeft": {"name": "ちっ左", "panel": MORPH_LIP, "ratio": 1},
    "mouthLowerDown": {"name": "むっ", "panel": MORPH_LIP, "ratio": 1, "binds": ["mouthLowerDownRight", "mouthLowerDownLeft"]},
    "mouthLowerDownRight": {"name": "むっ右", "panel": MORPH_LIP, "ratio": 1},
    "mouthLowerDownLeft": {"name": "むっ左", "panel": MORPH_LIP, "ratio": 1},
    "mouthStretch": {"name": "ぎりっ", "panel": MORPH_LIP, "ratio": 1, "binds": ["mouthStretchRight", "mouthStretchLeft"]},
    "mouthStretchRight": {"name": "ぎりっ右", "panel": MORPH_LIP, "ratio": 1},
    "mouthStretchLeft": {"name": "ぎりっ左", "panel": MORPH_LIP, "ratio": 1},
    "tongueOut": {"name": "べー", "panel": MORPH_LIP, "ratio": 1},
    "_mouthFunnel+SharpenLips": {"name": "うほっ", "panel": MORPH_LIP, "ratio": 1},
    "_mouthPress+CatMouth": {"name": "ω口", "panel": MORPH_LIP, "ratio": 1},
    "_mouthPress+CatMouth-ex": {"name": "ω口2", "panel": MORPH_LIP, "ratio": 1},
    "_mouthPress+DuckMouth": {"name": "ω口3", "panel": MORPH_LIP, "ratio": 1},
    "cheekPuff": {"name": "ぷくー", "panel": MORPH_LIP, "ratio": 1},
    "cheekPuff_R": {"name": "ぷくー右", "panel": MORPH_LIP, "ratio": 1, "split": "cheekPuff"},
    "cheekPuff_L": {"name": "ぷくー左", "panel": MORPH_LIP, "ratio": 1, "split": "cheekPuff"},

    "Fcl_MTH_SkinFung": {"name": "肌牙", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_MTH_SkinFung_L": {"name": "肌牙左", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_MTH_SkinFung_R": {"name": "肌牙右", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Fung1": {"name": "牙", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Fung1_Up": {"name": "牙上", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Fung1_Up_R": {"name": "牙上右", "panel": MORPH_LIP, "ratio": 1, "split": "Fcl_HA_Fung1_Up"},
    "Fcl_HA_Fung1_Up_L": {"name": "牙上左", "panel": MORPH_LIP, "ratio": 1, "split": "Fcl_HA_Fung1_Up"},
    "Fcl_HA_Fung1_Low": {"name": "牙下", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Fung1_Low_R": {"name": "牙下右", "panel": MORPH_LIP, "ratio": 1, "split": "Fcl_HA_Fung1_Low"},
    "Fcl_HA_Fung1_Low_L": {"name": "牙下左", "panel": MORPH_LIP, "ratio": 1, "split": "Fcl_HA_Fung1_Low"},
    "Fcl_HA_Fung2": {"name": "ギザ歯", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Fung2_Up": {"name": "ギザ歯上", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Fung2_Low": {"name": "ギザ歯下", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Fung3": {"name": "真ん中牙", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Fung3_Up": {"name": "真ん中牙上", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Fung3_Low": {"name": "真ん中牙下", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Hide": {"name": "歯隠", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Short": {"name": "歯短", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Short_Up": {"name": "歯短上", "panel": MORPH_LIP, "ratio": 1},
    "Fcl_HA_Short_Low": {"name": "歯短下", "panel": MORPH_LIP, "ratio": 1},

    "Fcl_ALL_Neutral": {"name": "ニュートラル", "panel": MORPH_OTHER},
    "Fcl_ALL_Angry": {"name": "怒", "panel": MORPH_OTHER},
    "Fcl_ALL_Fun": {"name": "楽", "panel": MORPH_OTHER},
    "Fcl_ALL_Joy": {"name": "喜", "panel": MORPH_OTHER},
    "Fcl_ALL_Sorrow": {"name": "哀", "panel": MORPH_OTHER},
    "Fcl_ALL_Surprised": {"name": "驚", "panel": MORPH_OTHER},
}
