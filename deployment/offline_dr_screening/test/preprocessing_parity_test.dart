import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:offirst_dr_screening/model_preprocessing.dart';

void main() {
  test('RGB PNG yields normalized NCHW float32', () {
    final image = img.Image(width: 1, height: 1)
      ..setPixelRgb(0, 0, 255, 128, 0);
    final tensor = prepareFundusTensor(img.encodePng(image));
    const plane = 512 * 512;
    expect(tensor, hasLength(3 * plane));
    expect(tensor[0], closeTo(2.2489083, 1e-6));
    expect(tensor[plane], closeTo(0.2051822, 1e-6));
    expect(tensor.last, closeTo(-1.8044444, 1e-6));
  });

  final fixtureDirectory = Platform.environment['OFFIRST_PARITY_DIR'];
  test('production Dart tensors match the shared Pillow goldens', () {
    final dir = fixtureDirectory!;
    final jpeg = File('$dir/normal-420.jpg').readAsBytesSync();
    expect(
        () => prepareFundusTensor(Uint8List.fromList([0xff, 0xd8, 0xff, 0xd9])),
        throwsFormatException);
    expect(
        () => prepareFundusTensor(
            Uint8List.sublistView(jpeg, 0, jpeg.length ~/ 2)),
        throwsFormatException);
    final fixtures =
        jsonDecode(File('$dir/fixtures.json').readAsStringSync()) as List;
    final mismatches = <String>[];
    for (final row in fixtures) {
      final name = row['name'] as String;
      final actual = prepareFundusTensor(File('$dir/$name').readAsBytesSync());
      final bytes = ByteData(actual.lengthInBytes);
      for (var i = 0; i < actual.length; i++) {
        bytes.setFloat32(i * 4, actual[i], Endian.little);
      }
      File('$dir/$name.dart.f32').writeAsBytesSync(bytes.buffer.asUint8List());
      final golden = File('$dir/$name.pillow.f32').readAsBytesSync();
      if (golden.length != bytes.lengthInBytes) {
        mismatches.add('$name: different tensor length');
        continue;
      }
      final reference = ByteData.sublistView(golden);
      var differences = 0;
      for (var offset = 0; offset < golden.length; offset += 4) {
        if (bytes.getUint32(offset, Endian.little) !=
            reference.getUint32(offset, Endian.little)) {
          differences++;
        }
      }
      if (differences != 0) {
        mismatches.add('$name: $differences different float32 values');
      }
    }
    expect(mismatches, isEmpty);
  }, skip: fixtureDirectory == null);
}
