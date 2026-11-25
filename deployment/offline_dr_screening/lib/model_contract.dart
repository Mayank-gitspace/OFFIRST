import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:onnxruntime/onnxruntime.dart';
// The pinned wrapper exposes names, but not type and shape inspection.
// ignore: implementation_imports
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as ort;

const modelInput = 'fundus_image';
const modelOutput = 'dr_probability';
const modelInputShape = [1, 3, 512, 512];

ort.OrtApi get _api => OrtEnv.instance.ortApiPtr.ref;

void validateModelContract(OrtSession session) {
  if (session.inputNames.length != 1 ||
      session.inputNames.single != modelInput ||
      session.outputNames.length != 1 ||
      session.outputNames.single != modelOutput) {
    throw StateError('Unexpected ONNX model inputs or outputs.');
  }

  _validateSessionTensor(session, input: true, dimensions: [-1, 3, 512, 512]);
  _validateSessionTensor(session, input: false, dimensions: [-1, 1]);
}

void _validateSessionTensor(OrtSession session,
    {required bool input, required List<int> dimensions}) {
  using((arena) {
    final info = arena<Pointer<ort.OrtTypeInfo>>();
    final read =
        input ? _api.SessionGetInputTypeInfo : _api.SessionGetOutputTypeInfo;
    OrtStatus.checkOrtStatus(read.asFunction<
            ort.OrtStatusPtr Function(Pointer<ort.OrtSession>, int,
                Pointer<Pointer<ort.OrtTypeInfo>>)>()(
        Pointer<ort.OrtSession>.fromAddress(session.address), 0, info));
    try {
      final tensor = arena<Pointer<ort.OrtTensorTypeAndShapeInfo>>();
      OrtStatus.checkOrtStatus(_api.CastTypeInfoToTensorInfo.asFunction<
              ort.OrtStatusPtr Function(Pointer<ort.OrtTypeInfo>,
                  Pointer<Pointer<ort.OrtTensorTypeAndShapeInfo>>)>()(
          info.value, tensor));
      _validateTensor(
          tensor.value, dimensions, arena, input ? modelInput : modelOutput);
    } finally {
      // The cast only borrows the tensor information from the parent type.
      _api.ReleaseTypeInfo.asFunction<
          void Function(Pointer<ort.OrtTypeInfo>)>()(info.value);
    }
  });
}

double readModelProbability(OrtValueTensor tensor) {
  return using((arena) {
    final value = Pointer<ort.OrtValue>.fromAddress(tensor.address);
    final info = arena<Pointer<ort.OrtTensorTypeAndShapeInfo>>();
    OrtStatus.checkOrtStatus(_api.GetTensorTypeAndShape.asFunction<
        ort.OrtStatusPtr Function(Pointer<ort.OrtValue>,
            Pointer<Pointer<ort.OrtTensorTypeAndShapeInfo>>)>()(value, info));
    try {
      _validateTensor(info.value, const [1, 1], arena, modelOutput);
      final data = arena<Pointer<Void>>();
      OrtStatus.checkOrtStatus(_api.GetTensorMutableData.asFunction<
          ort.OrtStatusPtr Function(
              Pointer<ort.OrtValue>, Pointer<Pointer<Void>>)>()(value, data));
      final probability = data.value.cast<Float>().value;
      if (!probability.isFinite || probability < 0 || probability > 1) {
        throw StateError('Model returned an invalid probability.');
      }
      return probability;
    } finally {
      _api.ReleaseTensorTypeAndShapeInfo.asFunction<
          void Function(Pointer<ort.OrtTensorTypeAndShapeInfo>)>()(info.value);
    }
  });
}

void _validateTensor(Pointer<ort.OrtTensorTypeAndShapeInfo> info,
    List<int> dimensions, Arena arena, String name) {
  if (info == nullptr) throw StateError('$name is not a tensor.');
  final type = arena<Int32>();
  final rank = arena<Size>();
  OrtStatus.checkOrtStatus(_api.GetTensorElementType.asFunction<
      ort.OrtStatusPtr Function(Pointer<ort.OrtTensorTypeAndShapeInfo>,
          Pointer<Int32>)>()(info, type));
  OrtStatus.checkOrtStatus(_api.GetDimensionsCount.asFunction<
      ort.OrtStatusPtr Function(Pointer<ort.OrtTensorTypeAndShapeInfo>,
          Pointer<Size>)>()(info, rank));
  if (type.value !=
          ort.ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      rank.value != dimensions.length) {
    throw StateError('Unexpected element type or rank for $name.');
  }
  final values = arena<Int64>(dimensions.length);
  OrtStatus.checkOrtStatus(_api.GetDimensions.asFunction<
      ort.OrtStatusPtr Function(Pointer<ort.OrtTensorTypeAndShapeInfo>,
          Pointer<Int64>, int)>()(info, values, dimensions.length));
  for (var axis = 0; axis < dimensions.length; axis++) {
    if (values[axis] != dimensions[axis]) {
      throw StateError('Unexpected dimension for $name at axis $axis.');
    }
  }
}
