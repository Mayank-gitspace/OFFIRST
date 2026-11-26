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

  test('startup removes only marked, unreferenced visit transactions',
      () async {
    const marker = '.offirst-pending-visit';
    final patient = PatientRecord(id: 'patient', name: 'Demo', createdAt: date);
    await store.addPatient(patient);
    final source = File('${root.path}/source')..writeAsBytesSync([42]);
    final completed = VisitRecord(id: recordId('VIS'), date: date, images: [
      ImageRecord(
          id: recordId('IMG'),
          eye: Eye.right,
          capturedAt: date,
          path: source.path,
          probability: .2,
          threshold: .48,
          modelVersion: 'test'),
    ]);
    await store.addVisit(patient.id, completed);
    final private = await store.directory;
    final visits = '${private.path}/visits';
    final staged = Directory('$visits/.staged-interrupted')..createSync();
    final renamed = Directory('$visits/renamed-interrupted')..createSync();
    final older = Directory('$visits/unmarked-older-visit')..createSync();
    final protectedByPath = Directory('$visits/referenced-by-path')
      ..createSync();
    File('${staged.path}/$marker').writeAsStringSync('staged-interrupted');
    File('${renamed.path}/$marker').writeAsStringSync('renamed-interrupted');
    File('${older.path}/original').writeAsBytesSync([11]);
    File('${protectedByPath.path}/$marker')
        .writeAsStringSync('referenced-by-path');
    final legacyOriginal = File('${protectedByPath.path}/original')
      ..writeAsBytesSync([12]);
    final committedMarker = File('$visits/${completed.id}/$marker')
      ..writeAsStringSync(completed.id);
    final metadata = File('${private.path}/offirst_patients.json');
    final records = jsonDecode(metadata.readAsStringSync()) as List;
    (records.single['visits'] as List).add({
      'date': date.toIso8601String(),
      'eye': 'OS',
      'imagePath': legacyOriginal.path,
      'probability': .2,
    });
    metadata.writeAsStringSync(jsonEncode(records));
    final before = metadata.readAsBytesSync();

    final restarted = PatientStore(directory: () async => private);
    final loaded = await restarted.load();
    expect(loaded.single.visits, hasLength(2));
    expect(metadata.readAsBytesSync(), before);
    expect(staged.existsSync(), isFalse);
    expect(renamed.existsSync(), isFalse);
    expect(committedMarker.existsSync(), isFalse);
    expect(File('${older.path}/original').readAsBytesSync(), [11]);
    expect(legacyOriginal.readAsBytesSync(), [12]);
    expect(
        (await restarted.original(loaded.single.visits.first.images.single))
            .readAsBytesSync(),
        [42]);
  });

  test('startup leaves marked images alone if metadata is missing or corrupt',
      () async {
    final private = await store.directory;
    final pending = Directory('${private.path}/visits/interrupted')
      ..createSync(recursive: true);
    File('${pending.path}/.offirst-pending-visit')
        .writeAsStringSync('interrupted');
    final image = File('${pending.path}/original')..writeAsBytesSync([7]);
    expect(await store.load(), isEmpty);
    expect(image.readAsBytesSync(), [7]);

    File('${private.path}/offirst_patients.json').writeAsStringSync('{');
    await expectLater(store.load(), throwsFormatException);
    expect(image.readAsBytesSync(), [7]);
  });

  for (final failure in ['second image copy', 'metadata write']) {
    test('$failure rolls back the visit and preserves previous visits',
        () async {
      final patient =
          PatientRecord(id: recordId('OFF'), name: 'Demo', createdAt: date);
      await store.addPatient(patient);
      final directory = await store.directory;
      final source = File('${root.path}/source')..writeAsBytesSync([42]);
      ImageRecord image(Eye eye, String path) => ImageRecord(
            id: recordId('IMG'),
            eye: eye,
            capturedAt: date,
            path: path,
            probability: .2,
            threshold: .48,
            modelVersion: 'test',
          );
      final completed = VisitRecord(
          id: recordId('VIS'),
          date: date,
          images: [image(Eye.right, source.path)]);
      await store.addVisit(patient.id, completed);
      final metadata = File('${directory.path}/offirst_patients.json');
      final previous = await metadata.readAsBytes();
      final obstruction = Directory('${metadata.path}.staged');
      if (failure == 'metadata write') await obstruction.create();
      final missing = File('${root.path}/missing');
      final pending = VisitRecord(id: recordId('VIS'), date: date, images: [
        image(Eye.right, source.path),
        image(Eye.left,
            failure == 'second image copy' ? missing.path : source.path),
      ]);

      await expectLater(store.addVisit(patient.id, pending),
          throwsA(isA<FileSystemException>()));
      expect(await metadata.readAsBytes(), previous);
      final visitDirectories =
          await Directory('${directory.path}/visits').list().toList();
      expect(
          visitDirectories
              .map((entry) => entry.path.split(Platform.pathSeparator).last),
          [completed.id]);
      final stored = (await store.load()).single.visits.single.images.single;
      expect(await (await store.original(stored)).readAsBytes(), [42]);

      if (await obstruction.exists()) await obstruction.delete();
      await missing.writeAsBytes([17]);
      await store.addVisit(patient.id, pending);
      expect((await store.load()).single.visits.map((visit) => visit.id),
          [pending.id, completed.id]);
    });
  }

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
            id: recordId('VIS'),
            date: date,
            images: [image(Eye.right, .2)],
            unavailableEyes: {Eye.left}));
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
    final unilateral = visits.singleWhere((visit) => visit.id != bilateral.id);
    expect(unilateral.forEye(Eye.right), hasLength(1));
    expect(unilateral.forEye(Eye.left), isEmpty);
    expect(unilateral.unavailableEyes, {Eye.left});
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
    expect((await store.load()).single.expertIds, isEmpty);
    expect(result.eye, Eye.left);
    expect(result.probability, .61);
    file.writeAsStringSync('invalid');
    await expectLater(
        store.addPatient(
            PatientRecord(id: 'new', name: 'Demo', createdAt: date)),
        throwsFormatException);
    expect(file.readAsStringSync(), 'invalid');
  });

  test('patient edits preserve IDs and visits; deletion isolates owned visits',
      () async {
    final first = PatientRecord(
        id: recordId('OFF'), name: 'First', createdAt: date, age: 42);
    final second =
        PatientRecord(id: recordId('OFF'), name: 'Second', createdAt: date);
    await store.addPatient(first);
    await store.addPatient(second);
    final source = File('${root.path}/image')..writeAsBytesSync([4, 5, 6]);
    final firstVisit = VisitRecord(id: recordId('VIS'), date: date, images: [
      ImageRecord(
          id: recordId('IMG'),
          eye: Eye.right,
          capturedAt: date,
          path: source.path,
          probability: .4,
          threshold: .48,
          modelVersion: 'test'),
    ]);
    final secondVisit = VisitRecord(id: recordId('VIS'), date: date, images: [
      ImageRecord(
          id: recordId('IMG'),
          eye: Eye.left,
          capturedAt: date,
          path: source.path,
          probability: .6,
          threshold: .48,
          modelVersion: 'test'),
    ]);
    await store.addVisit(first.id, firstVisit);
    await store.addVisit(second.id, secondVisit);
    final reportFolder =
        Directory('${(await store.directory).path}/screening_reports')
          ..createSync(recursive: true);
    final ownedReport =
        File('${reportFolder.path}/OFFIRST-${firstVisit.id}-v2.pdf')
          ..writeAsBytesSync([1]);
    await store.updatePatient(PatientRecord(
        id: first.id,
        name: 'First edited',
        createdAt: DateTime.utc(2000),
        age: 43,
        notes: 'Local note'));
    var rows = await store.load();
    final edited = rows.singleWhere((p) => p.id == first.id);
    expect(edited.name, 'First edited');
    expect(edited.createdAt, date);
    expect(edited.visits.single.id, firstVisit.id);
    expect(edited.age, 43);
    expect(edited.notes, 'Local note');

    await store.deletePatient(first.id);
    rows = await store.load();
    expect(rows.map((p) => p.id), [second.id]);
    expect(
        Directory('${(await store.directory).path}/visits/${firstVisit.id}')
            .existsSync(),
        isFalse);
    expect(ownedReport.existsSync(), isFalse);
    final retained = rows.single.visits.single.images.single;
    expect(await (await store.original(retained)).readAsBytes(), [4, 5, 6]);
  });

  test('patient identifiers cannot collide', () async {
    final patient = PatientRecord(id: 'OFF-fixed', name: 'A', createdAt: date);
    await store.addPatient(patient);
    await expectLater(
        store.addPatient(
            PatientRecord(id: patient.id, name: 'B', createdAt: date)),
        throwsStateError);
    expect((await store.load()).map((p) => p.name), ['A']);
  });

  test('expert records and patient associations persist and resolve safely',
      () async {
    final patient =
        PatientRecord(id: recordId('OFF'), name: 'Demo', createdAt: date);
    final expert = ExpertRecord(
        id: recordId('EXP'),
        name: 'Locally entered clinician',
        updatedAt: date,
        specialty: 'Ophthalmology');
    await store.addPatient(patient);
    await store.addExpert(expert);
    await store.setPatientExperts(patient.id, [expert.id]);
    await store.updatePatient(PatientRecord(
        id: patient.id,
        name: 'Demo updated',
        createdAt: date,
        notes: 'Updated'));
    expect((await store.load()).single.expertIds, [expert.id]);
    expect((await store.loadExperts()).single.id, expert.id);

    await store.deleteExpert(expert.id);
    expect((await store.loadExperts()), isEmpty);
    final rows = await store.load();
    expect(rows.single.name, 'Demo updated');
    expect(rows.single.expertIds, isEmpty);
  });
}
