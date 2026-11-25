import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'patient_store.dart';
import 'screening_service.dart';

class ScreeningPage extends StatefulWidget {
  const ScreeningPage(
      {super.key,
      required this.store,
      required this.patients,
      required this.onSaved});
  final PatientStore store;
  final Future<List<PatientRecord>> patients;
  final VoidCallback onSaved;

  @override
  State<ScreeningPage> createState() => _ScreeningPageState();
}

class _ScreeningPageState extends State<ScreeningPage> {
  final _picker = ImagePicker();
  final _service = ScreeningService();
  final _images = <ImageRecord>[];
  String _visitId = recordId('VIS');
  String? _patientId;
  Eye _eye = Eye.right;
  String? _message;
  bool _busy = false;

  Future<void> _add({required bool camera}) async {
    final eye = _eye;
    setState(() {
      _busy = true;
      _message = null;
    });
    try {
      final List<XFile> files;
      if (camera) {
        final photo = await _picker.pickImage(source: ImageSource.camera);
        files = [if (photo != null) photo];
      } else {
        files = await _picker.pickMultiImage();
      }
      var failures = 0;
      for (final photo in files) {
        if (!mounted) break;
        try {
          final result = await _service.predict(File(photo.path));
          if (!mounted) break;
          setState(() => _images.add(ImageRecord(
                id: recordId('IMG'),
                eye: eye,
                capturedAt: DateTime.now().toUtc(),
                path: photo.path,
                probability: result.probability,
                threshold: screeningThreshold,
                modelVersion: screeningModelVersion,
              )));
        } catch (_) {
          failures++;
        }
      }
      if (mounted && failures > 0) {
        setState(() => _message =
            '$failures image(s) could not be screened. Check the files or try again if the model failed to load.');
      }
    } catch (_) {
      if (mounted) {
        setState(() =>
            _message = 'Could not open the image source. Please try again.');
      }
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      } else {
        _service.dispose();
      }
    }
  }

  Future<void> _save() async {
    final patientId = _patientId;
    if (patientId == null || _images.isEmpty || _busy) return;
    setState(() {
      _busy = true;
      _message = null;
    });
    try {
      await widget.store.addVisit(
          patientId,
          VisitRecord(
            id: _visitId,
            date: DateTime.now().toUtc(),
            images: _images,
          ));
      if (!mounted) return;
      widget.onSaved();
      setState(() {
        _images.clear();
        _visitId = recordId('VIS');
        _message =
            'Visit saved. Open Patients to view the visit or create a PDF.';
      });
    } catch (_) {
      if (mounted) {
        setState(() => _message =
            'Could not save the visit. Your current selections are still here.');
      }
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      } else {
        _service.dispose();
      }
    }
  }

  @override
  void dispose() {
    if (!_busy) _service.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Text('Retinal image screening',
                style: Theme.of(context).textTheme.headlineSmall,
                softWrap: true),
            const Text('Experimental DR screening runs on this device.'),
            const SizedBox(height: 16),
            FutureBuilder<List<PatientRecord>>(
              future: widget.patients,
              builder: (context, snapshot) {
                if (snapshot.connectionState != ConnectionState.done) {
                  return const LinearProgressIndicator();
                }
                if (snapshot.hasError) {
                  return const Text('Could not load patient records.');
                }
                final patients = snapshot.data ?? [];
                if (patients.isEmpty) {
                  return const Text(
                      'Register a patient to save a screening visit.');
                }
                return DropdownButtonFormField<String>(
                  isExpanded: true,
                  initialValue: _patientId,
                  decoration: const InputDecoration(labelText: 'Patient'),
                  items: patients
                      .map((p) => DropdownMenuItem(
                          value: p.id,
                          child: Text(p.name, overflow: TextOverflow.ellipsis)))
                      .toList(),
                  onChanged:
                      _busy ? null : (id) => setState(() => _patientId = id),
                );
              },
            ),
            const SizedBox(height: 16),
            SegmentedButton<Eye>(
              segments: Eye.values
                  .map((eye) => ButtonSegment(
                        value: eye,
                        label: Text('${eye.label} (${eye.code})'),
                      ))
                  .toList(),
              selected: {_eye},
              onSelectionChanged:
                  _busy ? null : (eyes) => setState(() => _eye = eyes.single),
            ),
            const SizedBox(height: 12),
            Wrap(spacing: 8, runSpacing: 8, children: [
              FilledButton.icon(
                  onPressed: _busy ? null : () => _add(camera: false),
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text('Add fundus images')),
              OutlinedButton.icon(
                  onPressed: _busy ? null : () => _add(camera: true),
                  icon: const Icon(Icons.camera_alt_outlined),
                  label: const Text('Camera')),
            ]),
            if (_busy) const LinearProgressIndicator(),
            if (_message != null)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: Semantics(liveRegion: true, child: Text(_message!)),
              ),
            const SizedBox(height: 10),
            for (final eye in Eye.values)
              EyeResults(
                eye: eye,
                images: _images.where((image) => image.eye == eye),
                original: (image) async => File(image.path),
                onRemove: _busy
                    ? null
                    : (image) => setState(() => _images.remove(image)),
              ),
            const SizedBox(height: 12),
            FilledButton.tonal(
              onPressed:
                  _busy || _patientId == null || _images.isEmpty ? null : _save,
              child: const Text('Finalize and save visit'),
            ),
            const Padding(
                padding: EdgeInsets.only(top: 12),
                child: Text(screeningNotice)),
          ],
        ),
      );
}

const screeningNotice =
    'Experimental AI screening only. This output is not a medical diagnosis or a basis for treatment.';

class EyeResults extends StatelessWidget {
  const EyeResults(
      {super.key,
      required this.eye,
      required this.images,
      required this.original,
      this.onRemove});
  final Eye eye;
  final Iterable<ImageRecord> images;
  final Future<File> Function(ImageRecord) original;
  final ValueChanged<ImageRecord>? onRemove;

  @override
  Widget build(BuildContext context) => Card(
        child: Column(
          children: [
            ListTile(
              title: Text('${eye.label} (${eye.code})'),
              subtitle: Text('${images.length} screened image(s)'),
            ),
            for (final image in images)
              ListTile(
                leading: FutureBuilder<File>(
                  future: original(image),
                  builder: (context, snapshot) => snapshot.hasData
                      ? Image(
                          image: ResizeImage(FileImage(snapshot.data!),
                              width: 96,
                              height: 96,
                              policy: ResizeImagePolicy.fit),
                          width: 48,
                          height: 48,
                          fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) =>
                              const Icon(Icons.broken_image_outlined))
                      : const Icon(Icons.image_outlined),
                ),
                title: Text(
                    '${(image.probability * 100).toStringAsFixed(1)}% DR model probability'),
                subtitle: Text(image.flagged
                    ? 'At or above threshold'
                    : 'Below threshold'),
                trailing: onRemove == null
                    ? null
                    : IconButton(
                        tooltip: 'Remove image',
                        onPressed: () => onRemove!(image),
                        icon: const Icon(Icons.close),
                      ),
                onTap: () async {
                  final file = await original(image);
                  if (!context.mounted) return;
                  await showDialog<void>(
                      context: context,
                      builder: (context) => AlertDialog(
                            title: Text('${eye.label} image'),
                            content: Image(
                                image: ResizeImage(FileImage(file),
                                    width: 1200,
                                    height: 1200,
                                    policy: ResizeImagePolicy.fit),
                                fit: BoxFit.contain,
                                errorBuilder: (_, __, ___) =>
                                    const Text('Original image unavailable.')),
                            actions: [
                              TextButton(
                                  onPressed: () => Navigator.pop(context),
                                  child: const Text('Close'))
                            ],
                          ));
                },
              ),
          ],
        ),
      );
}
