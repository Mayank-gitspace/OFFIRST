import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:offirst_dr_screening/patient_store.dart';

void main() {
  late Directory root;
  late PatientStore store;
  final date = DateTime.utc(2026, 1, 1);

  setUp(() {
    root = Directory.systemTemp.createTempSync('offirst-store-');
    store =
        PatientStore(directory: () async => Directory('${root.path}/private'));
  });

  tearDown(() => root.deleteSync(recursive: true));

  test(
      'new store instances recover both eyes, all visits and copied original bytes',
      () async {
    final patient =
        PatientRecord(id: recordId('OFF'), name: 'Demo', createdAt: date);
    final other =
        PatientRecord(id: recordId('OFF'), name: 'Other', createdAt: date);
    await Future.wait([store.addPatient(patient), store.addPatient(other)]);
    final input = File('${root.path}/source')..writeAsBytesSync([1, 2, 3]);
    ImageRecord image(Eye eye, double score) => ImageRecord(
          id: recordId('IMG'),
          eye: eye,
          capturedAt: date,
          path: input.path,
          probability: score,
          threshold: .48,
          modelVersion: 'test',
        );
    final bilateral = VisitRecord(id: recordId('VIS'), date: date, images: [
      image(Eye.right, .48),
      image(Eye.right, .65),
      image(Eye.left, .3),
    ]);
    await store.addVisit(patient.id, bilateral);
    await store.addVisit(patient.id, bilateral);
    await store.addVisit(
        patient.id,
        VisitRecord(
            id: recordId('VIS'), date: date, images: [image(Eye.left, .2)]));
    input.deleteSync();

    final restarted =
        PatientStore(directory: () async => Directory('${root.path}/private'));
    final records = await restarted.load();
    expect(records, hasLength(2));
    final visits =
        records.singleWhere((record) => record.id == patient.id).visits;
    expect(visits, hasLength(2));
    final saved = visits.singleWhere((visit) => visit.id == bilateral.id);
    expect(saved.images.map((image) => image.id).toSet(), hasLength(3));
    expect(
        saved.forEye(Eye.right).map((image) => image.probability), [.48, .65]);
    expect(saved.forEye(Eye.left).single.flagged, isFalse);
    for (final image in saved.images) {
      expect(image.path, contains('/${image.eye.code}/'));
      expect(await (await restarted.original(image)).readAsBytes(), [1, 2, 3]);
    }
  });

  test(
      'retains legacy image results and refuses to overwrite unreadable history',
      () async {
    final directory = await store.directory;
    directory.createSync();
    final file = File('${directory.path}/offirst_patients.json');
    file.writeAsStringSync(jsonEncode([
      {
        'id': 'OFF-old',
        'name': 'Demo',
        'createdAt': date.toIso8601String(),
        'visits': [
          {
            'date': date.toIso8601String(),
            'eye': 'OS',
            'imagePath': '/old/path',
            'probability': .61
          }
        ],
      }
    ]));
    final result = (await store.load()).single.visits.single.images.single;
    expect(result.eye, Eye.left);
    expect(result.probability, .61);
    file.writeAsStringSync('invalid');
    await expectLater(
        store.addPatient(
            PatientRecord(id: 'new', name: 'Demo', createdAt: date)),
        throwsFormatException);
    expect(file.readAsStringSync(), 'invalid');
  });
}
