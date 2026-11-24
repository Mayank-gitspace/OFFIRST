import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

class RgbPixels {
  const RgbPixels(this.width, this.height, this.rgb);
  final int width;
  final int height;
  final Uint8List rgb;
}

typedef _DecodeNative = Pointer<Uint8> Function(
    Pointer<Uint8>, Size, Pointer<Int32>, Pointer<Int32>);
typedef _DecodeDart = Pointer<Uint8> Function(
    Pointer<Uint8>, int, Pointer<Int32>, Pointer<Int32>);
typedef _FreeNative = Void Function(Pointer<Uint8>);

final _jpeg = _openJpeg();
final _decode =
    _jpeg.lookupFunction<_DecodeNative, _DecodeDart>('offirst_decode_jpeg');
final _free = _jpeg.lookupFunction<_FreeNative, void Function(Pointer<Uint8>)>(
    'offirst_free_jpeg');

DynamicLibrary _openJpeg() {
  if (Platform.isAndroid) return DynamicLibrary.open('liboffirst_jpeg.so');
  if (Platform.isWindows) return DynamicLibrary.open('offirst_jpeg.dll');
  throw UnsupportedError(
      'Reference JPEG decoding is unavailable on this platform.');
}

/// Uses the same raw orientation, integer IDCT and chroma upsampling as Pillow.
RgbPixels decodeReferenceJpeg(Uint8List encoded) {
  return using((arena) {
    final data = arena<Uint8>(encoded.length);
    data.asTypedList(encoded.length).setAll(0, encoded);
    final width = arena<Int32>();
    final height = arena<Int32>();
    final rgb = _decode(data, encoded.length, width, height);
    if (rgb == nullptr) {
      throw const FormatException('Unsupported or invalid JPEG.');
    }
    try {
      return RgbPixels(width.value, height.value,
          Uint8List.fromList(rgb.asTypedList(width.value * height.value * 3)));
    } finally {
      _free(rgb);
    }
  });
}
