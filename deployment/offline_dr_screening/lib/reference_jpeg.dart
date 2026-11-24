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

/// Borrows decoded pixels for a synchronous operation. The view must not
/// escape [consume]. Only resized/normalized results should be returned.
T withReferenceJpeg<T>(Uint8List encoded, T Function(RgbPixels) consume) {
  final (rgb, width, height) = using((arena) {
    final data = arena<Uint8>(encoded.length);
    data.asTypedList(encoded.length).setAll(0, encoded);
    final width = arena<Int32>();
    final height = arena<Int32>();
    final rgb = _decode(data, encoded.length, width, height);
    if (rgb == nullptr) {
      throw const FormatException('JPEG is invalid or exceeds decoder limits.');
    }
    return (rgb, width.value, height.value);
  });
  // The decoder has finished reading the encoded copy. The arena above has
  // already freed it. Read native RGB directly without a second full image.
  try {
    return consume(
        RgbPixels(width, height, rgb.asTypedList(width * height * 3)));
  } finally {
    _free(rgb);
  }
}
