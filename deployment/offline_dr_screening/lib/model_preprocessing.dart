import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import 'reference_jpeg.dart';

const modelImageSize = 512;

final _normalization = _buildNormalization();

/// Mirrors deployment inference.py: raw encoded orientation, RGB conversion,
/// Pillow bilinear resize, and float32 normalization into planar RGB.
Float32List prepareFundusTensor(Uint8List encoded) {
  var pixels = decodeReferenceRgb(encoded);
  if (pixels.width != modelImageSize) {
    pixels = _resizeAxis(pixels, horizontal: true);
  }
  if (pixels.height != modelImageSize) {
    pixels = _resizeAxis(pixels, horizontal: false);
  }

  const plane = modelImageSize * modelImageSize;
  final result = Float32List(3 * plane);
  final rgb = pixels.rgb;
  for (var index = 0, offset = 0; index < plane; index++, offset += 3) {
    result[index] = _normalization[0][rgb[offset]];
    result[plane + index] = _normalization[1][rgb[offset + 1]];
    result[2 * plane + index] = _normalization[2][rgb[offset + 2]];
  }
  return result;
}

RgbPixels decodeReferenceRgb(Uint8List encoded) {
  if (encoded.length >= 2 && encoded[0] == 0xff && encoded[1] == 0xd8) {
    return decodeReferenceJpeg(encoded);
  }
  // Unlike the JPEG decoder, these Dart decoders leave pixel orientation alone.
  final decoded = img.decodeImage(encoded);
  if (decoded == null) {
    throw const FormatException('Unsupported or invalid image.');
  }
  // Pillow converts unsigned 16-bit grayscale (I;16) by saturating each
  // sample at 255. RGB/RGBA 16-bit PNGs instead decode to the upper byte.
  if (decoded.format == img.Format.uint16 && decoded.numChannels == 1) {
    final bytes = Uint8List(decoded.width * decoded.height * 3);
    var offset = 0;
    for (final pixel in decoded) {
      final value = pixel.r.toInt().clamp(0, 255);
      bytes[offset++] = value;
      bytes[offset++] = value;
      bytes[offset++] = value;
    }
    return RgbPixels(decoded.width, decoded.height, bytes);
  }
  final rgb = decoded.convert(format: img.Format.uint8, numChannels: 3);
  return RgbPixels(
      rgb.width, rgb.height, rgb.getBytes(order: img.ChannelOrder.rgb));
}

List<Float32List> _buildNormalization() {
  final means = Float32List.fromList([0.485, 0.456, 0.406]);
  final stds = Float32List.fromList([0.229, 0.224, 0.225]);
  final scratch = Float32List(1);
  return List.generate(3, (channel) {
    final values = Float32List(256);
    for (var byte = 0; byte < 256; byte++) {
      scratch[0] = byte / 255.0;
      scratch[0] = scratch[0] - means[channel];
      values[byte] = scratch[0] / stds[channel];
    }
    return values;
  });
}

const _weightBits = 22;
const _weightScale = 1 << _weightBits;
const _round = _weightScale >> 1;

List<(int, Int32List)> _taps(int input) {
  final scale = input / modelImageSize;
  // Widen the triangular filter when reducing size to remove high frequencies.
  final radius = math.max(1.0, scale);
  final inverse = 1.0 / radius;
  return List.generate(modelImageSize, (index) {
    final center = (index + 0.5) * scale;
    final first = math.max(0, (center - radius + 0.5).truncate());
    final last = math.min(input, (center + radius + 0.5).truncate());
    final weights = List<double>.generate(last - first, (tap) {
      final distance = ((first + tap - center + 0.5) * inverse).abs();
      return math.max(0.0, 1.0 - distance);
    });
    var total = 0.0;
    for (final weight in weights) {
      total += weight;
    }
    return (
      first,
      Int32List.fromList([
        for (final weight in weights)
          ((weight / total) * _weightScale + 0.5).truncate(),
      ])
    );
  });
}

RgbPixels _resizeAxis(RgbPixels image, {required bool horizontal}) {
  final width = horizontal ? modelImageSize : image.width;
  final height = horizontal ? image.height : modelImageSize;
  final taps = _taps(horizontal ? image.width : image.height);
  final stride = horizontal ? 3 : 3 * image.width;
  final output = Uint8List(width * height * 3);
  final source = image.rgb;

  for (var y = 0, outputOffset = 0; y < height; y++) {
    for (var x = 0; x < width; x++, outputOffset += 3) {
      final (first, weights) = taps[horizontal ? x : y];
      var offset =
          (horizontal ? y * image.width + first : first * image.width + x) * 3;
      var r = _round;
      var g = _round;
      var b = _round;
      for (final weight in weights) {
        r += source[offset] * weight;
        g += source[offset + 1] * weight;
        b += source[offset + 2] * weight;
        offset += stride;
      }
      // Pillow rounds each intermediate pass to uint8 before using the next.
      output[outputOffset] = (r >> _weightBits).clamp(0, 255);
      output[outputOffset + 1] = (g >> _weightBits).clamp(0, 255);
      output[outputOffset + 2] = (b >> _weightBits).clamp(0, 255);
    }
  }
  return RgbPixels(width, height, output);
}
